#!/usr/bin/env python3
"""python/explain_parameters.py

Generate a "model parameters" report for both Baseline and Proposed models.

In this project, the Siamese models are not supervised on explicit attributes
(e.g., ears/eyes/nose labels). Instead, we approximate the "parameters" the
models rely on by:

1) Computing saliency heatmaps (gradient-based) for the similarity score.
2) Aggregating saliency into interpretable regions (ears/eyes/muzzle/fur/body)
   inside the detected pet bounding box, plus background outside the bbox.
3) Comparing how these region attributions and classification metrics change
   under a lowlight scenario.

Outputs:
- cache/Parameters/report.json
- cache/Parameters/heatmaps/<scenario>/<model>/<id>_img1.png
- cache/Parameters/heatmaps/<scenario>/<model>/<id>_img2.png

Run:
  python python/explain_parameters.py --max-pairs 200 --examples-per-type 1

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

# YOLO (for pet bbox -> isolate background)
from ultralytics import YOLO


if TYPE_CHECKING:
    import tensorflow as tf

    Tensor = tf.Tensor
else:
    Tensor = Any


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "Parameters"
HEATMAP_DIR = CACHE_DIR / "heatmaps"
DEFAULT_DATA_ROOT = ROOT / "data"

LOWLIGHT_CONFIG = {
    "brightness": 0.35,
    "gamma": 1.8,
    "noise_std": 0.02,
}


class StatusWriter:
    def __init__(self, status_json: Path):
        self.status_json = status_json
        self._last_write = 0.0
        self._last_percent = -1
        self._last_stage = None
        self._last_state = None

    def write(self, payload: Dict[str, Any], *, throttle_s: float = 0.35) -> None:
        now = time.time()
        pct = int(payload.get("percent", -1))
        stage = payload.get("stage")
        state = payload.get("state")
        stage_changed = (stage != self._last_stage) or (state != self._last_state)
        if (now - self._last_write) < float(throttle_s) and pct == self._last_percent and not stage_changed:
            return
        self._last_write = now
        self._last_percent = pct
        self._last_stage = stage
        self._last_state = state

        payload = dict(payload)
        payload.setdefault("updated_at", _now_iso())
        self.status_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.status_json.with_suffix(self.status_json.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        tmp.replace(self.status_json)


class EmbeddingsDiskCache:
    """Persistent embedding cache using sqlite3.

    Stores float32 embeddings per image path, keyed by exact path string.
    Invalidates automatically if cache meta changes (model file / settings).
    """

    def __init__(self, db_path: Path, *, meta: Dict[str, str]):
        self.db_path = db_path
        self.meta = {str(k): str(v) for k, v in (meta or {}).items()}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.execute("PRAGMA synchronous=NORMAL;")
        self.con.execute("PRAGMA temp_store=MEMORY;")
        self._init_schema()
        self._ensure_meta()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS meta(
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings(
              path TEXT PRIMARY KEY,
              mtime_ns INTEGER NOT NULL,
              size_bytes INTEGER NOT NULL,
              dim INTEGER NOT NULL,
              vec BLOB NOT NULL
            );
            """
        )
        self.con.commit()

    def _ensure_meta(self) -> None:
        cur = self.con.execute("SELECT k, v FROM meta")
        existing = {str(k): str(v) for (k, v) in cur.fetchall()}
        if existing and existing != self.meta:
            # Model/settings changed: invalidate cache.
            self.con.execute("DELETE FROM embeddings")
            self.con.execute("DELETE FROM meta")
            self.con.executemany("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", list(self.meta.items()))
            self.con.commit()
        elif not existing:
            self.con.executemany("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", list(self.meta.items()))
            self.con.commit()

    def get_many(self, paths: List[Path]) -> Dict[str, np.ndarray]:
        if not paths:
            return {}

        # Query in chunks to avoid SQLite variable limits.
        out: Dict[str, np.ndarray] = {}
        strs = [str(p) for p in paths]
        chunk_size = 900
        for start in range(0, len(strs), chunk_size):
            chunk = strs[start : start + chunk_size]
            q = "SELECT path, mtime_ns, size_bytes, dim, vec FROM embeddings WHERE path IN (%s)" % (
                ",".join(["?"] * len(chunk))
            )
            rows = self.con.execute(q, chunk).fetchall()
            for path_s, mtime_ns, size_bytes, dim, vec in rows:
                try:
                    st = os.stat(path_s)
                except Exception:
                    continue
                if int(st.st_mtime_ns) != int(mtime_ns) or int(st.st_size) != int(size_bytes):
                    continue
                arr = np.frombuffer(vec, dtype=np.float32)
                arr = arr.reshape((int(dim),)).astype(np.float32, copy=False)
                out[str(path_s)] = arr
        return out

    def put_many(self, embeddings: Dict[str, np.ndarray]) -> None:
        if not embeddings:
            return
        rows = []
        for path_s, emb in embeddings.items():
            try:
                st = os.stat(path_s)
            except Exception:
                continue
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    str(path_s),
                    int(st.st_mtime_ns),
                    int(st.st_size),
                    int(emb.shape[0]),
                    sqlite3.Binary(emb.tobytes(order="C")),
                )
            )
        if not rows:
            return
        self.con.executemany(
            "INSERT OR REPLACE INTO embeddings(path, mtime_ns, size_bytes, dim, vec) VALUES(?,?,?,?,?)",
            rows,
        )
        self.con.commit()


def _model_file_meta(model_path: Path) -> Dict[str, str]:
    try:
        st = model_path.stat()
        return {
            "model_path": _safe_rel(model_path),
            "model_mtime_ns": str(int(st.st_mtime_ns)),
            "model_size_bytes": str(int(st.st_size)),
        }
    except Exception:
        return {"model_path": _safe_rel(model_path)}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _json_sanitize(obj: Any) -> Any:
    """Recursively convert objects into JSON-safe primitives.

    Key point: replace non-finite floats (NaN/±Inf) with None so that JSON is
    RFC-compliant and can be parsed by PHP's json_decode.
    """

    if obj is None:
        return None

    if isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None

    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    return obj


def _load_image_224(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr


def _apply_lowlight(
    arr01: np.ndarray,
    *,
    brightness: float = float(LOWLIGHT_CONFIG["brightness"]),
    gamma: float = float(LOWLIGHT_CONFIG["gamma"]),
    noise_std: float = float(LOWLIGHT_CONFIG["noise_std"]),
    noise_seed: Optional[int] = None,
) -> np.ndarray:
    """Simple lowlight scenario.

    - Reduce brightness
    - Apply gamma curve
    - Add small gaussian noise

    Input/Output are float32 in [0,1].
    """
    x = np.clip(arr01, 0.0, 1.0).astype(np.float32)
    x = np.power(x, gamma)
    x = x * brightness
    if noise_std > 0:
        if noise_seed is None:
            noise = np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        else:
            rng = np.random.default_rng(int(noise_seed) & 0xFFFFFFFF)
            noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        x = x + noise
    return np.clip(x, 0.0, 1.0)


def _lowlight_seed_for_path(*, base_seed: int, path: Path, which: int) -> int:
    """Deterministic seed for lowlight noise based on image identity.

    This keeps lowlight consistent for the same image even if it appears in
    multiple pairs (important for caching embeddings).

    which: 1 for img1, 2 for img2.
    """
    key = f"{_safe_rel(path)}|{int(which)}".encode("utf-8", errors="ignore")
    h = zlib.crc32(key) & 0xFFFFFFFF
    return (int(base_seed) * 1000003 + int(h)) & 0xFFFFFFFF


def _resolve_image_path(raw: str, *, data_root: Path) -> Path:
    s = (raw or "").strip().strip('"').strip("'")
    s = s.replace("/", os.sep).replace("\\", os.sep)
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p

    # Most CSV paths are like: (Preprocessed) breed_organized_with_images\...\val\...
    cand1 = (data_root / p)
    if cand1.exists():
        return cand1

    cand2 = (ROOT / p)
    if cand2.exists():
        return cand2

    # As a last resort, try stripping a leading data root name
    if s.lower().startswith("data" + os.sep):
        cand3 = ROOT / s
        if cand3.exists():
            return cand3

    return cand1


def _load_pairs_from_csv(csv_path: Path, *, data_root: Path, max_pairs: int, seed: int) -> Tuple[List[Tuple[Path, Path]], np.ndarray, Dict[str, Any]]:
    """Load pairs + labels from a CSV like data/processed_data/val_pairs.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    pairs: List[Tuple[Path, Path]] = []
    y: List[int] = []
    meta: Dict[str, Any] = {
        "csv": _safe_rel(csv_path),
        "data_root": _safe_rel(data_root),
        "label_strategy": "CSV label column (0/1) where 1 means similar.",
    }

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row")

        required = {"img1_path", "img2_path", "label"}
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            p1 = _resolve_image_path(row.get("img1_path", ""), data_root=data_root)
            p2 = _resolve_image_path(row.get("img2_path", ""), data_root=data_root)
            try:
                lbl = int(str(row.get("label", "0")).strip())
            except Exception:
                lbl = 0
            pairs.append((p1, p2))
            y.append(1 if lbl != 0 else 0)

    if not pairs:
        raise RuntimeError("CSV produced 0 pairs")

    # Shuffle consistently then cap
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    if max_pairs > 0:
        idx = idx[: min(max_pairs, len(idx))]
    pairs = [pairs[i] for i in idx]
    y_arr = np.asarray([y[i] for i in idx], dtype=np.int32)
    return pairs, y_arr, meta


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w)),
            y2=max(0, min(self.y2, h)),
        )


class PetDetector:
    """Detect dog/cat bbox using YOLOv8n COCO classes."""

    # COCO indices: 15=cat, 16=dog
    CAT_CLASS = 15
    DOG_CLASS = 16

    def __init__(self, weights_path: Path):
        self.model = YOLO(str(weights_path))

    def detect_bbox(self, img01_224: np.ndarray) -> Optional[BBox]:
        """Return bbox in 224x224 image coords."""
        img_uint8 = (np.clip(img01_224, 0.0, 1.0) * 255.0).astype(np.uint8)
        results = self.model.predict(img_uint8, verbose=False)
        if not results:
            return None
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None

        best = None
        best_conf = -1.0
        for b in r0.boxes:
            cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            if cls not in (self.CAT_CLASS, self.DOG_CLASS):
                continue
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            if conf > best_conf:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                best = BBox(x1, y1, x2, y2)
                best_conf = conf
        if best is None:
            return None
        return best.clamp(224, 224)


def _overlay_heatmap(img01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a heatmap onto RGB image (both in [0,1]) and return uint8 RGB."""
    import cv2

    h = np.clip(heat01, 0.0, 1.0)
    heat_u8 = (h * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heat_color01 = heat_color.astype(np.float32) / 255.0

    out = (1.0 - alpha) * img01 + alpha * heat_color01
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _saliency_pairs_batch(sim_fn_batch, imgs1_01: np.ndarray, imgs2_01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradient saliency for a batch of pairs.

    Returns:
      sal1: (N,224,224) float32 in [0,1]
      sal2: (N,224,224) float32 in [0,1]
      scores: (N,) float32
    """
    import tensorflow as tf

    x1 = tf.convert_to_tensor(imgs1_01, dtype=tf.float32)
    x2 = tf.convert_to_tensor(imgs2_01, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x1)
        tape.watch(x2)
        s = sim_fn_batch(x1, x2)  # (N,)

    g1 = tape.gradient(s, x1)
    g2 = tape.gradient(s, x2)
    del tape

    def _to_sal_batch(g: Optional[tf.Tensor]) -> np.ndarray:
        if g is None:
            n = int(imgs1_01.shape[0])
            return np.zeros((n, 224, 224), dtype=np.float32)
        sal = tf.reduce_mean(tf.abs(g), axis=-1)  # (N,224,224)
        sal_np = sal.numpy().astype(np.float32)
        mn = np.min(sal_np, axis=(1, 2), keepdims=True)
        mx = np.max(sal_np, axis=(1, 2), keepdims=True)
        sal_np = (sal_np - mn) / (mx - mn + 1e-8)
        return sal_np

    sal1 = _to_sal_batch(g1)
    sal2 = _to_sal_batch(g2)
    scores = np.asarray(s.numpy(), dtype=np.float32)
    return sal1, sal2, scores


def _saliency_to_regions(sal01: np.ndarray, bbox: Optional[BBox]) -> Dict[str, float]:
    """Aggregate saliency into interpretable "parameters".

    If bbox is present, background is outside bbox.
    Zones inside bbox (top->bottom): ears, eyes, muzzle, fur_body.

    Returns percentages summing to ~100.
    """
    h, w = sal01.shape
    total = float(np.sum(sal01) + 1e-8)

    # background mask
    bg = np.zeros((h, w), dtype=bool)
    if bbox is not None:
        bg[:, :] = True
        bg[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] = False

    def region_sum(mask: np.ndarray) -> float:
        return float(np.sum(sal01[mask]))

    regions: Dict[str, float] = {}
    regions["background"] = 100.0 * (region_sum(bg) / total) if bbox is not None else 0.0

    if bbox is None:
        # Fallback: approximate within whole image
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

    box_h = max(1, y2 - y1)

    def _band(y_start_frac: float, y_end_frac: float) -> np.ndarray:
        ys = y1 + int(round(y_start_frac * box_h))
        ye = y1 + int(round(y_end_frac * box_h))
        ys = max(y1, min(ys, y2))
        ye = max(y1, min(ye, y2))
        mask = np.zeros((h, w), dtype=bool)
        if ye > ys and x2 > x1:
            mask[ys:ye, x1:x2] = True
        return mask

    ears_m = _band(0.0, 0.20)
    eyes_m = _band(0.20, 0.40)
    muzzle_m = _band(0.40, 0.60)
    fur_m = _band(0.60, 1.0)

    regions["ears_region"] = 100.0 * (region_sum(ears_m) / total)
    regions["eyes_region"] = 100.0 * (region_sum(eyes_m) / total)
    regions["muzzle_region"] = 100.0 * (region_sum(muzzle_m) / total)
    regions["fur_body_region"] = 100.0 * (region_sum(fur_m) / total)

    # Normalize small numeric drift
    s = sum(regions.values())
    if s > 0:
        for k in list(regions.keys()):
            regions[k] = regions[k] * (100.0 / s)

    return regions


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _metrics_from_counts(c: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def _best_threshold(
    scores01: np.ndarray,
    y_true: np.ndarray,
    *,
    optimize: str = "f1",
    beta: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Pick threshold in [0,1] maximizing the chosen objective.

    optimize: one of {'f1','accuracy','precision','recall','f_beta'}
    beta: used only when optimize='f_beta' (beta>1 favors recall -> fewer FN).
    """
    best_t = 0.5
    best_m = {"f1": -1.0}
    best_obj = -1.0

    opt = (optimize or "f1").strip().lower()
    b = float(beta) if beta is not None else 1.0
    b = 1.0 if not np.isfinite(b) or b <= 0 else b

    # Candidate thresholds based on unique scores (plus endpoints)
    uniq = np.unique(np.clip(scores01, 0.0, 1.0))
    if uniq.size > 500:
        # downsample for speed
        idx = np.linspace(0, uniq.size - 1, 500).astype(int)
        uniq = uniq[idx]
    candidates = np.unique(np.concatenate([uniq, np.array([0.0, 1.0], dtype=np.float32)]))

    for t in candidates:
        pred = (scores01 >= t).astype(np.int32)
        c = _confusion_counts(y_true, pred)
        m = _metrics_from_counts(c)

        if opt == "accuracy":
            obj = float(m.get("accuracy", 0.0))
        elif opt == "precision":
            obj = float(m.get("precision", 0.0))
        elif opt == "recall":
            obj = float(m.get("recall", 0.0))
        elif opt == "f_beta":
            p = float(m.get("precision", 0.0))
            r = float(m.get("recall", 0.0))
            obj = (1.0 + b * b) * p * r / max(1e-12, (b * b) * p + r)
        else:
            obj = float(m.get("f1", 0.0))

        # Prefer higher objective; tie-breaker prefers higher recall (fewer FN)
        if (obj > best_obj) or (obj == best_obj and float(m.get("recall", 0.0)) > float(best_m.get("recall", 0.0))):
            best_obj = obj
            best_m = m
            best_t = float(t)

    return best_t, best_m


def _extract_label_from_filename(p: Path) -> str:
    """Heuristic label from filenames like Breed_Breed_123_xxx.jpg or name_uuid.jpg."""
    stem = p.stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def _gather_images(pre_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    out: List[Path] = []

    for sub in ["Cats", "Dogs"]:
        d = pre_dir / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)

    return out


def _make_pairs(paths: List[Path], *, max_pairs: int, seed: int) -> Tuple[List[Tuple[Path, Path]], np.ndarray]:
    """Create balanced positive/negative pairs using filename-derived labels."""
    rng = random.Random(seed)
    by_label: Dict[str, List[Path]] = {}
    for p in paths:
        lbl = _extract_label_from_filename(p)
        by_label.setdefault(lbl, []).append(p)

    labels = [k for k, v in by_label.items() if len(v) >= 2]
    if len(labels) < 2:
        raise RuntimeError("Not enough labels with >=2 images to create positive pairs")

    pos: List[Tuple[Path, Path]] = []
    neg: List[Tuple[Path, Path]] = []

    target_each = max_pairs // 2

    # positives
    while len(pos) < target_each:
        lbl = rng.choice(labels)
        a, b = rng.sample(by_label[lbl], 2)
        pos.append((a, b))

    # negatives
    all_labels = list(by_label.keys())
    while len(neg) < target_each:
        lbl1, lbl2 = rng.sample(all_labels, 2)
        a = rng.choice(by_label[lbl1])
        b = rng.choice(by_label[lbl2])
        neg.append((a, b))

    pairs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    # shuffle in unison
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    pairs = [pairs[i] for i in idx]
    y = y[idx]
    return pairs, y


class BaselineEmbedder:
    def __init__(self, model_path: Path):
        from compute_matches_baseline import PetMatcher

        self.matcher = PetMatcher(model_path=str(model_path))
        if self.matcher.base_network is None:
            raise RuntimeError("Baseline: could not extract base network for embeddings")
        self.model = self.matcher.base_network

    @property
    def param_count(self) -> int:
        return int(self.matcher.model.count_params()) if self.matcher.model is not None else int(self.model.count_params())

    def embed(self, img01: Tensor) -> Tensor:
        import tensorflow as tf
        # img01: (1,224,224,3) float32 in [0,1]
        out = self.model(img01, training=False)
        out = tf.reshape(out, [out.shape[0], -1])
        return out

    def embed_batch(self, imgs01: Tensor) -> Tensor:
        import tensorflow as tf
        out = self.model(imgs01, training=False)
        out = tf.reshape(out, [out.shape[0], -1])
        return out

    def similarity01(self, img1: Tensor, img2: Tensor) -> Tensor:
        import tensorflow as tf
        e1 = self.embed(img1)
        e2 = self.embed(img2)
        d = tf.norm(e1 - e2, axis=1)  # (1,)
        s = tf.exp(-d)
        return s[0]

    def similarity01_batch(self, imgs1: Tensor, imgs2: Tensor) -> Tensor:
        import tensorflow as tf
        e1 = self.embed_batch(imgs1)
        e2 = self.embed_batch(imgs2)
        d = tf.norm(e1 - e2, axis=1)
        return tf.exp(-d)


class ProposedEmbedder:
    def __init__(self, model_path: Path, *, use_mnetv2: bool = True, mnetv2_weight: float = 0.4):
        from compute_matches import PetMatchingEngine

        import tensorflow as tf  # noqa: F401

        self.engine = PetMatchingEngine(
            str(model_path),
            debug=False,
            batch_size=16,
            num_workers=1,
            use_mnetv2=use_mnetv2,
            mnetv2_weight=mnetv2_weight,
        )
        self.engine.load_model()
        if use_mnetv2:
            try:
                self.engine.load_mnetv2_model()
            except Exception:
                self.engine.use_mnetv2 = False

    @property
    def param_count(self) -> int:
        if self.engine.model is not None:
            return int(self.engine.model.count_params())
        if self.engine.mnetv2_model is not None:
            return int(self.engine.mnetv2_model.count_params())
        return 0

    def _custom_embedding(self, img01: Tensor) -> Tensor:
        import tensorflow as tf
        if self.engine.embedding_model is None:
            # Fall back to MobileNetV2-only
            return self._mnetv2_embedding(img01)

        out = self.engine.embedding_model(img01, training=False)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            out = out[1]
        out = tf.reshape(out, [out.shape[0], -1])
        return out

    def _mnetv2_embedding(self, img01: Tensor) -> Tensor:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnetv2_preprocess
        if not self.engine.use_mnetv2 or self.engine.mnetv2_model is None:
            raise RuntimeError("Proposed: MobileNetV2 is disabled/unavailable")
        x255 = img01 * 255.0
        x = mnetv2_preprocess(x255)
        out = self.engine.mnetv2_model(x, training=False)
        out = tf.reshape(out, [out.shape[0], -1])
        return out

    def embed(self, img01: Tensor) -> Tensor:
        import tensorflow as tf
        # Mirror compute_matches.py get_combined_embedding behavior
        custom = self._custom_embedding(img01)

        if self.engine.embedding_model is None:
            # already fallback
            emb = tf.nn.l2_normalize(custom, axis=1)
            return emb

        if not self.engine.use_mnetv2 or self.engine.mnetv2_model is None:
            return tf.nn.l2_normalize(custom, axis=1)

        mnet = self._mnetv2_embedding(img01)

        w = float(getattr(self.engine, "mnetv2_weight", 0.4))
        w = max(0.0, min(1.0, w))

        custom_n = tf.nn.l2_normalize(custom, axis=1)
        mnet_n = tf.nn.l2_normalize(mnet, axis=1)

        # Weight and concat, then normalize
        combined = tf.concat([(1.0 - w) * custom_n, w * mnet_n], axis=1)
        combined = tf.nn.l2_normalize(combined, axis=1)
        return combined

    def embed_batch(self, imgs01: Tensor) -> Tensor:
        # embed() already supports batch if we pass a batch tensor.
        return self.embed(imgs01)

    def similarity01(self, img1: Tensor, img2: Tensor) -> Tensor:
        import tensorflow as tf
        e1 = self.embed(img1)
        e2 = self.embed(img2)
        cos = tf.reduce_sum(e1 * e2, axis=1)  # (1,)
        # Match compute_matches.py cosine_distance():
        # distance = clip(1 - cos, 0, 1)  => similarity = 1 - distance = clip(cos, 0, 1)
        s01 = tf.clip_by_value(cos, 0.0, 1.0)
        return s01[0]

    def similarity01_batch(self, imgs1: Tensor, imgs2: Tensor) -> Tensor:
        import tensorflow as tf
        e1 = self.embed_batch(imgs1)
        e2 = self.embed_batch(imgs2)
        cos = tf.reduce_sum(e1 * e2, axis=1)
        return tf.clip_by_value(cos, 0.0, 1.0)


def _unique_paths_from_pairs(pairs: List[Tuple[Path, Path]]) -> List[Path]:
    seen: Dict[str, Path] = {}
    for p1, p2 in pairs:
        seen[str(p1)] = p1
        seen[str(p2)] = p2
    return list(seen.values())


def _compute_embeddings_cache(
    paths: List[Path],
    *,
    embedder,
    batch_size: int,
    lowlight: bool,
    seed: int,
    io_workers: int,
    disk_cache: Optional[EmbeddingsDiskCache],
    status_cb: Optional[Callable[[Dict[str, Any]], None]],
    status_context: Dict[str, Any],
    progress_done_ref: Optional[List[int]],
) -> Dict[str, np.ndarray]:
    import tensorflow as tf

    out: Dict[str, np.ndarray] = {}

    def _load_one(p: Path) -> np.ndarray:
        a = _load_image_224(p)
        if lowlight:
            a = _apply_lowlight(a, noise_seed=_lowlight_seed_for_path(base_seed=seed, path=p, which=1))
        return a

    total = int(len(paths))
    cached_count = 0
    if disk_cache is not None:
        cached = disk_cache.get_many(paths)
        out.update(cached)
        cached_count = int(len(cached))
        if status_cb is not None:
            done0 = cached_count
            if progress_done_ref is not None and progress_done_ref:
                progress_done_ref[0] += done0
            status_cb(
                {
                    **status_context,
                    "state": "running",
                    "stage": "embeddings",
                    "done": done0,
                    "total": total,
                    "cached": cached_count,
                    "overall_embeddings_done": int(progress_done_ref[0]) if (progress_done_ref is not None and progress_done_ref) else int(status_context.get("overall_embeddings_done", 0)),
                }
            )

    missing = [p for p in paths if str(p) not in out]
    if not missing:
        return out

    for start in range(0, len(missing), max(1, batch_size)):
        chunk = missing[start : start + max(1, batch_size)]

        if int(io_workers) > 1:
            with ThreadPoolExecutor(max_workers=int(io_workers)) as ex:
                imgs = list(ex.map(_load_one, chunk))
        else:
            imgs = [_load_one(p) for p in chunk]

        x = tf.convert_to_tensor(np.stack(imgs, axis=0), dtype=tf.float32)
        emb = embedder.embed_batch(x)
        emb_np = emb.numpy().astype(np.float32)
        new_rows: Dict[str, np.ndarray] = {}
        for i, p in enumerate(chunk):
            out[str(p)] = emb_np[i]
            new_rows[str(p)] = emb_np[i]
        if disk_cache is not None:
            disk_cache.put_many(new_rows)

        if status_cb is not None:
            done = int(cached_count + min(len(missing), start + len(chunk)))
            if progress_done_ref is not None and progress_done_ref:
                progress_done_ref[0] += int(len(chunk))
            status_cb(
                {
                    **status_context,
                    "state": "running",
                    "stage": "embeddings",
                    "done": done,
                    "total": total,
                    "cached": cached_count,
                    "overall_embeddings_done": int(progress_done_ref[0]) if (progress_done_ref is not None and progress_done_ref) else int(status_context.get("overall_embeddings_done", 0)),
                }
            )
    return out


def _saliency_pair(sim_fn, img1_01: np.ndarray, img2_01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    import tensorflow as tf

    x1 = tf.convert_to_tensor(img1_01[None, ...], dtype=tf.float32)
    x2 = tf.convert_to_tensor(img2_01[None, ...], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x1)
        tape.watch(x2)
        s = sim_fn(x1, x2)

    g1 = tape.gradient(s, x1)
    g2 = tape.gradient(s, x2)
    del tape

    def _to_sal(g: tf.Tensor) -> np.ndarray:
        if g is None:
            return np.zeros((224, 224), dtype=np.float32)
        sal = tf.reduce_mean(tf.abs(g), axis=-1)[0]
        sal = sal.numpy().astype(np.float32)
        mn = float(np.min(sal))
        mx = float(np.max(sal))
        sal = (sal - mn) / (mx - mn + 1e-8)
        return sal

    return _to_sal(g1), _to_sal(g2), float(s.numpy())


def _roc_curve_auc(scores: np.ndarray, y_true: np.ndarray, *, max_points: int = 400) -> Dict[str, Any]:
    """Compute ROC curve (FPR/TPR) and AUC.

    - scores: higher means more likely positive (same pet)
    - y_true: 0/1 labels (1 == positive)
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)
    if scores.ndim != 1 or y_true.ndim != 1 or scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must be 1D arrays of same length")

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        # Degenerate ROC: cannot define both TPR and FPR.
        return {
            "auc": 0.0,
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0],
            # Use nulls instead of ±Infinity to keep JSON valid
            "thresholds": [None, None],
        }

    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    # Indices where threshold (score) changes
    distinct = np.where(np.diff(s) != 0.0)[0]
    thresh_idx = np.r_[distinct, len(s) - 1]

    tpr = tps[thresh_idx] / float(pos)
    fpr = fps[thresh_idx] / float(neg)

    # Add endpoints
    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    thresholds = np.r_[np.inf, s[thresh_idx], -np.inf]

    auc = float(np.trapz(tpr, fpr))

    # Downsample for JSON size / UI rendering
    if max_points is not None and int(max_points) > 0 and len(fpr) > int(max_points):
        k = int(max_points)
        idx = np.unique(np.round(np.linspace(0, len(fpr) - 1, k)).astype(np.int64))
        fpr = fpr[idx]
        tpr = tpr[idx]
        thresholds = thresholds[idx]

    return {
        "auc": auc,
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) if np.isfinite(x) else None for x in thresholds],
    }


def _pick_example_indices(scores: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, *, per_type: int) -> Dict[str, List[int]]:
    """Pick indices for TP/TN/FP/FN, choosing the most confident ones."""
    idxs: Dict[str, List[int]] = {"TP": [], "TN": [], "FP": [], "FN": []}

    # sort by confidence away from threshold (0.5 placeholder), using score extremes
    order = np.argsort(-scores)  # high->low
    for i in order:
        if y_true[i] == 1 and y_pred[i] == 1 and len(idxs["TP"]) < per_type:
            idxs["TP"].append(int(i))
        elif y_true[i] == 0 and y_pred[i] == 0 and len(idxs["TN"]) < per_type:
            idxs["TN"].append(int(i))
        elif y_true[i] == 0 and y_pred[i] == 1 and len(idxs["FP"]) < per_type:
            idxs["FP"].append(int(i))
        elif y_true[i] == 1 and y_pred[i] == 0 and len(idxs["FN"]) < per_type:
            idxs["FN"].append(int(i))
        if all(len(v) >= per_type for v in idxs.values()):
            break

    return idxs


def run_report(
    *,
    preprocessed_dir: Path,
    pairs_csv: Optional[Path],
    data_root: Path,
    baseline_model: Path,
    proposed_model: Path,
    yolov8_weights: Path,
    max_pairs: int,
    seed: int,
    examples_per_type: int,
    out_json: Path,
    heatmap_prefix: str,
    batch_size: int,
    region_sample_n: int,
    saliency_batch: int,
    tf_intra_threads: int,
    tf_inter_threads: int,
    io_workers: int,
    emb_cache_dir: Optional[Path],
    status_json: Optional[Path],
    proposed_mnetv2_weight: float = 0.4,
    proposed_threshold_optimize: str = "f1",
    proposed_threshold_beta: float = 1.0,
) -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    if int(tf_intra_threads) > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(tf_intra_threads))
        except Exception:
            pass
    if int(tf_inter_threads) > 0:
        try:
            tf.config.threading.set_inter_op_parallelism_threads(int(tf_inter_threads))
        except Exception:
            pass
    np.random.seed(seed)
    random.seed(seed)

    status = StatusWriter(status_json) if status_json is not None else None

    def _status_update(fields: Dict[str, Any]) -> None:
        if status is None:
            return
        # Map embedding & saliency progress into a single percent.
        # Embeddings are the long pole: allocate 0-85% for embeddings, 85-100% for saliency/writing.
        emb_done = int(fields.get("overall_embeddings_done", 0))
        emb_total = int(fields.get("overall_embeddings_total", 1))
        sal_done = int(fields.get("overall_saliency_done", 0))
        sal_total = int(fields.get("overall_saliency_total", 1))

        emb_pct = 0.0 if emb_total <= 0 else float(emb_done) / float(max(1, emb_total))
        sal_pct = 0.0 if sal_total <= 0 else float(sal_done) / float(max(1, sal_total))

        stage = fields.get("stage")
        if stage == "embeddings":
            pct = int(round(85.0 * emb_pct))
        elif stage in ("saliency", "writing"):
            pct = int(round(85.0 + 15.0 * sal_pct))
        elif stage == "done":
            pct = 100
        else:
            pct = int(round(85.0 * emb_pct))

        # Never show 100% unless done.
        if stage != "done" and pct >= 100:
            pct = 99

        payload = dict(fields)
        payload["percent"] = max(0, min(100, int(pct)))
        status.write(payload)

    dataset_meta: Dict[str, Any]
    if pairs_csv is not None:
        pairs, y_true, dataset_meta = _load_pairs_from_csv(pairs_csv, data_root=data_root, max_pairs=max_pairs, seed=seed)
    else:
        images = _gather_images(preprocessed_dir)
        if len(images) < 20:
            raise RuntimeError(f"Not enough images found under {preprocessed_dir}")
        pairs, y_true = _make_pairs(images, max_pairs=max_pairs, seed=seed)
        dataset_meta = {
            "root": _safe_rel(preprocessed_dir),
            "label_strategy": "Filename-derived label (prefix before first underscore). Similar=Same label; Dissimilar=Different label.",
        }

    detector = PetDetector(yolov8_weights)

    # IMPORTANT:
    # compute_matches.py sets TensorFlow threading config at import time.
    # That config must happen BEFORE TensorFlow initializes. Loading the baseline
    # .h5 model can initialize TF, so we create ProposedEmbedder first.
    proposed = ProposedEmbedder(
        proposed_model,
        use_mnetv2=True,
        mnetv2_weight=float(proposed_mnetv2_weight),
    )
    baseline = BaselineEmbedder(baseline_model)

    if status is not None:
        _status_update(
            {
                "state": "running",
                "stage": "starting",
                "message": "Initializing models and dataset...",
                "overall_embeddings_done": 0,
                "overall_embeddings_total": 1,
                "overall_saliency_done": 0,
                "overall_saliency_total": 1,
            }
        )

    heatmap_root = HEATMAP_DIR / heatmap_prefix if str(heatmap_prefix).strip() else HEATMAP_DIR
    heatmap_root.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "models": {
            "baseline": {
                "path": _safe_rel(baseline_model),
                "type": "Siamese CNN (MobileNetV2)",
                "param_count": baseline.param_count,
                "input": [224, 224, 3],
            },
            "proposed": {
                "path": _safe_rel(proposed_model),
                "type": "Siamese Capsule Network + Attention + MobileNetV2 Ensemble",
                "param_count": proposed.param_count,
                "input": [224, 224, 3],
                "mnetv2_weight": float(getattr(proposed.engine, "mnetv2_weight", 0.4)),
                "use_mnetv2": bool(getattr(proposed.engine, "use_mnetv2", True)),
            },
        },
        "dataset": {
            **dataset_meta,
            "pair_count": int(len(pairs)),
        },
        "parameters_definition": [
            {
                "name": "Ears Region",
                "key": "ears_region",
                "definition": "Top 20% of detected pet bounding box (proxy for ears/head top).",
            },
            {
                "name": "Eyes Region",
                "key": "eyes_region",
                "definition": "Next 20% band inside the pet bbox (proxy for eyes/upper face).",
            },
            {
                "name": "Muzzle/Snout Region",
                "key": "muzzle_region",
                "definition": "Middle 20% band inside the pet bbox (proxy for nose/snout).",
            },
            {
                "name": "Fur/Body Region",
                "key": "fur_body_region",
                "definition": "Lower 40% inside the pet bbox (proxy for coat texture/pattern/body shape).",
            },
            {
                "name": "Background",
                "key": "background",
                "definition": "Pixels outside detected pet bbox (proxy for background reliance / spurious cues).",
            },
        ],
        "scenarios": {},
        "lowlight_config": dict(LOWLIGHT_CONFIG),
        "performance_factors": [
            "Image quality (blur, motion, resolution)",
            "Lighting and exposure (including lowlight)",
            "Pose/angle and occlusion (face partially visible)",
            "Background leakage (model attention outside the pet)",
            "Threshold selection for same/different decision",
            "Dataset imbalance per breed/label and domain shift vs real uploads",
        ],
        "gaps_in_parameters_context": [
            "No supervised annotations for explicit parts (ears/eyes/nose). The models learn implicit features; we approximate them using saliency + region aggregation.",
            "Label proxy (breed/name) may differ from real "
            "same-pet identity matching; this analysis uses available dataset structure.",
            "BBox is from generic COCO detector; part regions are coarse bands (not true keypoint detection).",
        ],
        "architectural_challenges": [
            "Siamese similarity relies on thresholding; optimal threshold can drift by scenario (e.g., lowlight).",
            "Capsule/attention models can be harder to interpret than plain CNNs; saliency helps but remains approximate.",
            "Ensembling (custom + MobileNetV2) improves robustness but adds sensitivity to preprocessing differences between branches.",
        ],
    }

    uniq_all = _unique_paths_from_pairs(pairs)
    total_embedding_passes = 4  # normal/lowlight x baseline/proposed
    overall_embeddings_total = int(len(uniq_all) * total_embedding_passes)
    overall_embeddings_done_ref = [0]

    # Rough total saliency pair budget: per scenario, per model.
    # (examples up to 4 types * examples_per_type) + region_sample_n
    approx_examples_per_model = int(4 * max(0, int(examples_per_type)))
    overall_saliency_total = int(2 * 2 * (max(0, int(region_sample_n)) + approx_examples_per_model))
    overall_saliency_done_ref = [0]

    def _make_disk_cache(*, scenario: str, model_key: str, model_path: Path, extra_meta: Dict[str, str]) -> Optional[EmbeddingsDiskCache]:
        if emb_cache_dir is None:
            return None
        key = str(heatmap_prefix).strip() or "default"
        db = emb_cache_dir / f"{key}_{scenario}_{model_key}.sqlite3"
        meta = {
            **_model_file_meta(model_path),
            **{str(k): str(v) for k, v in (extra_meta or {}).items()},
            "scenario": str(scenario),
            "model_key": str(model_key),
            "seed": str(int(seed)),
            "lowlight": "1" if scenario == "lowlight" else "0",
            "lowlight_config": json.dumps(LOWLIGHT_CONFIG, sort_keys=True),
        }
        return EmbeddingsDiskCache(db, meta=meta)

    def _scenario_run(name: str, lowlight: bool) -> Dict[str, Any]:
        # Fast path: embed each unique image once, then score all pairs via numpy.
        uniq = uniq_all

        scenario_name = "lowlight" if lowlight else "normal"

        cache_b = _make_disk_cache(
            scenario=scenario_name,
            model_key="baseline",
            model_path=baseline_model,
            extra_meta={"embedder": "baseline"},
        )
        cache_p = _make_disk_cache(
            scenario=scenario_name,
            model_key="proposed",
            model_path=proposed_model,
            extra_meta={
                "embedder": "proposed",
                "use_mnetv2": str(bool(getattr(proposed.engine, "use_mnetv2", True))),
                "mnetv2_weight": str(float(getattr(proposed.engine, "mnetv2_weight", 0.4))),
            },
        )

        try:
            emb_b = _compute_embeddings_cache(
                uniq,
                embedder=baseline,
                batch_size=batch_size,
                lowlight=lowlight,
                seed=seed,
                io_workers=io_workers,
                disk_cache=cache_b,
                status_cb=_status_update,
                status_context={
                    "scenario": scenario_name,
                    "model": "baseline",
                    "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
                    "overall_embeddings_total": int(overall_embeddings_total),
                    "overall_saliency_done": int(overall_saliency_done_ref[0]),
                    "overall_saliency_total": int(overall_saliency_total),
                },
                progress_done_ref=overall_embeddings_done_ref,
            )
        finally:
            if cache_b is not None:
                cache_b.close()

        try:
            emb_p = _compute_embeddings_cache(
                uniq,
                embedder=proposed,
                batch_size=batch_size,
                lowlight=lowlight,
                seed=seed,
                io_workers=io_workers,
                disk_cache=cache_p,
                status_cb=_status_update,
                status_context={
                    "scenario": scenario_name,
                    "model": "proposed",
                    "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
                    "overall_embeddings_total": int(overall_embeddings_total),
                    "overall_saliency_done": int(overall_saliency_done_ref[0]),
                    "overall_saliency_total": int(overall_saliency_total),
                },
                progress_done_ref=overall_embeddings_done_ref,
            )
        finally:
            if cache_p is not None:
                cache_p.close()

        e1_b = np.stack([emb_b[str(p1)] for (p1, _p2) in pairs], axis=0)
        e2_b = np.stack([emb_b[str(p2)] for (_p1, p2) in pairs], axis=0)
        d = np.linalg.norm(e1_b - e2_b, axis=1)
        scores_b_arr = np.exp(-d).astype(np.float32)

        e1_p = np.stack([emb_p[str(p1)] for (p1, _p2) in pairs], axis=0)
        e2_p = np.stack([emb_p[str(p2)] for (_p1, p2) in pairs], axis=0)
        cos = np.sum(e1_p * e2_p, axis=1)
        # IMPORTANT: must match compute_matches.py cosine_distance():
        # distance = clip(1 - cos, 0, 1) => similarity = 1 - distance = clip(cos, 0, 1)
        scores_p_arr = np.clip(cos, 0.0, 1.0).astype(np.float32)

        roc_b = _roc_curve_auc(scores_b_arr, y_true)
        roc_p = _roc_curve_auc(scores_p_arr, y_true)

        t_b, m_b = _best_threshold(scores_b_arr, y_true, optimize="f1")
        t_p, m_p = _best_threshold(
            scores_p_arr,
            y_true,
            optimize=str(proposed_threshold_optimize),
            beta=float(proposed_threshold_beta),
        )

        y_pred_b = (scores_b_arr >= t_b).astype(np.int32)
        y_pred_p = (scores_p_arr >= t_p).astype(np.int32)

        c_b = _confusion_counts(y_true, y_pred_b)
        c_p = _confusion_counts(y_true, y_pred_p)

        examples: Dict[str, Any] = {"baseline": [], "proposed": []}

        # Pick indices per confusion type for each model
        idxs_b = _pick_example_indices(scores_b_arr, y_true, y_pred_b, per_type=examples_per_type)
        idxs_p = _pick_example_indices(scores_p_arr, y_true, y_pred_p, per_type=examples_per_type)

        bbox_cache: Dict[str, Optional[BBox]] = {}

        def _get_bbox(img01: np.ndarray, *, cache_key: str) -> Optional[BBox]:
            if cache_key in bbox_cache:
                return bbox_cache[cache_key]
            bb = detector.detect_bbox(img01)
            bbox_cache[cache_key] = bb
            return bb

        def _render_examples(model_name: str, idx_map: Dict[str, List[int]], sim_fn, sim_fn_batch, threshold: float):
            out_ex = []

            # Flatten selection into a list so we can batch saliency.
            selection: List[Tuple[str, int]] = []
            for kind, ids in idx_map.items():
                for i in ids:
                    selection.append((kind, int(i)))

            if not selection:
                return out_ex

            out_dir = heatmap_root / name / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Process in mini-batches for faster gradient computation.
            sb = max(1, int(saliency_batch))
            for start in range(0, len(selection), sb):
                chunk = selection[start : start + sb]
                imgs1: List[np.ndarray] = []
                imgs2: List[np.ndarray] = []
                paths: List[Tuple[Path, Path]] = []
                kinds: List[str] = []
                idxs: List[int] = []
                for kind, i in chunk:
                    p1, p2 = pairs[i]
                    img1 = _load_image_224(p1)
                    img2 = _load_image_224(p2)
                    if lowlight:
                        img1 = _apply_lowlight(img1, noise_seed=_lowlight_seed_for_path(base_seed=seed, path=p1, which=1))
                        img2 = _apply_lowlight(img2, noise_seed=_lowlight_seed_for_path(base_seed=seed, path=p2, which=2))
                    imgs1.append(img1)
                    imgs2.append(img2)
                    paths.append((p1, p2))
                    kinds.append(kind)
                    idxs.append(i)

                sal1_b, sal2_b, scores_b = _saliency_pairs_batch(sim_fn_batch, np.stack(imgs1, axis=0), np.stack(imgs2, axis=0))

                # Progress update for saliency
                overall_saliency_done_ref[0] += int(len(chunk))
                _status_update(
                    {
                        "state": "running",
                        "stage": "saliency",
                        "scenario": scenario_name,
                        "model": str(model_name),
                        "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
                        "overall_embeddings_total": int(overall_embeddings_total),
                        "overall_saliency_done": int(overall_saliency_done_ref[0]),
                        "overall_saliency_total": int(overall_saliency_total),
                        "message": "Generating heatmaps/examples...",
                    }
                )

                for j in range(len(chunk)):
                    kind = kinds[j]
                    i = idxs[j]
                    p1, p2 = paths[j]
                    img1 = imgs1[j]
                    img2 = imgs2[j]
                    sal1 = sal1_b[j]
                    sal2 = sal2_b[j]
                    score = float(scores_b[j])

                    bbox1 = _get_bbox(img1, cache_key=f"{name}|{_safe_rel(p1)}")
                    bbox2 = _get_bbox(img2, cache_key=f"{name}|{_safe_rel(p2)}")
                    reg1 = _saliency_to_regions(sal1, bbox1)
                    reg2 = _saliency_to_regions(sal2, bbox2)

                    label_same = bool(int(y_true[i]) == 1)
                    pred_same = bool(score >= threshold)
                    if label_same and pred_same:
                        kind_actual = "TP"
                    elif (not label_same) and (not pred_same):
                        kind_actual = "TN"
                    elif (not label_same) and pred_same:
                        kind_actual = "FP"
                    else:
                        kind_actual = "FN"

                    ex_id = f"{kind.lower()}_{i:04d}"
                    out1 = _overlay_heatmap(img1, sal1)
                    out2 = _overlay_heatmap(img2, sal2)

                    f1 = out_dir / f"{ex_id}_img1.png"
                    f2 = out_dir / f"{ex_id}_img2.png"
                    Image.fromarray(out1).save(f1)
                    Image.fromarray(out2).save(f2)

                    out_ex.append(
                        {
                            "kind": kind_actual,
                            "kind_expected": kind,
                            "pair_index": int(i),
                            "label_same": label_same,
                            "score": float(score),
                            "threshold": float(threshold),
                            "pred_same": pred_same,
                            "img1": _safe_rel(p1),
                            "img2": _safe_rel(p2),
                            "heatmap1": _safe_rel(f1),
                            "heatmap2": _safe_rel(f2),
                            "bbox1": None if bbox1 is None else {"x1": bbox1.x1, "y1": bbox1.y1, "x2": bbox1.x2, "y2": bbox1.y2},
                            "bbox2": None if bbox2 is None else {"x1": bbox2.x1, "y1": bbox2.y1, "x2": bbox2.x2, "y2": bbox2.y2},
                            "regions_img1": reg1,
                            "regions_img2": reg2,
                        }
                    )
            return out_ex

        examples["baseline"] = _render_examples("baseline", idxs_b, baseline.similarity01, baseline.similarity01_batch, t_b)
        examples["proposed"] = _render_examples("proposed", idxs_p, proposed.similarity01, proposed.similarity01_batch, t_p)

        # Aggregate region attribution across all pairs (fast approximation)
        # We do it on a small subset to keep runtime reasonable.
        sample_n = min(max(0, int(region_sample_n)), len(pairs))
        if sample_n <= 0:
            sample_n = 0
            region_avg_b = {"ears_region": 0.0, "eyes_region": 0.0, "muzzle_region": 0.0, "fur_body_region": 0.0, "background": 0.0}
            region_avg_p = {"ears_region": 0.0, "eyes_region": 0.0, "muzzle_region": 0.0, "fur_body_region": 0.0, "background": 0.0}
            return {
                "thresholds": {
                    "baseline": {"best_threshold": float(t_b), "metrics_at_best": m_b, "confusion": c_b},
                    "proposed": {"best_threshold": float(t_p), "metrics_at_best": m_p, "confusion": c_p},
                },
                "roc": {
                    "baseline": roc_b,
                    "proposed": roc_p,
                },
                "region_attribution_avg_percent": {
                    "baseline": region_avg_b,
                    "proposed": region_avg_p,
                },
                "examples": examples,
            }

        sample_idx = np.random.choice(len(pairs), size=sample_n, replace=False)

        def _region_summary(sim_fn_batch) -> Dict[str, float]:
            sums = {"ears_region": 0.0, "eyes_region": 0.0, "muzzle_region": 0.0, "fur_body_region": 0.0, "background": 0.0}
            sb = max(1, int(saliency_batch))

            sample_list = [int(i) for i in sample_idx]
            for start in range(0, len(sample_list), sb):
                chunk = sample_list[start : start + sb]
                imgs1: List[np.ndarray] = []
                imgs2: List[np.ndarray] = []
                p1s: List[Path] = []
                for i in chunk:
                    p1, p2 = pairs[i]
                    img1 = _load_image_224(p1)
                    img2 = _load_image_224(p2)
                    if lowlight:
                        img1 = _apply_lowlight(img1, noise_seed=_lowlight_seed_for_path(base_seed=seed, path=p1, which=1))
                        img2 = _apply_lowlight(img2, noise_seed=_lowlight_seed_for_path(base_seed=seed, path=p2, which=2))
                    imgs1.append(img1)
                    imgs2.append(img2)
                    p1s.append(p1)

                sal1_b, _sal2_b, _scores = _saliency_pairs_batch(sim_fn_batch, np.stack(imgs1, axis=0), np.stack(imgs2, axis=0))

                overall_saliency_done_ref[0] += int(len(chunk))
                _status_update(
                    {
                        "state": "running",
                        "stage": "saliency",
                        "scenario": scenario_name,
                        "model": "region_summary",
                        "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
                        "overall_embeddings_total": int(overall_embeddings_total),
                        "overall_saliency_done": int(overall_saliency_done_ref[0]),
                        "overall_saliency_total": int(overall_saliency_total),
                        "message": "Computing region attribution averages...",
                    }
                )
                for j, i in enumerate(chunk):
                    p1 = p1s[j]
                    img1 = imgs1[j]
                    bbox1 = _get_bbox(img1, cache_key=f"{name}|{_safe_rel(p1)}")
                    reg = _saliency_to_regions(sal1_b[j], bbox1)
                    for k in sums:
                        sums[k] += float(reg.get(k, 0.0))

            for k in sums:
                sums[k] /= float(max(1, sample_n))
            return sums

        region_avg_b = _region_summary(baseline.similarity01_batch)
        region_avg_p = _region_summary(proposed.similarity01_batch)

        return {
            "thresholds": {
                "baseline": {"best_threshold": float(t_b), "metrics_at_best": m_b, "confusion": c_b},
                "proposed": {"best_threshold": float(t_p), "metrics_at_best": m_p, "confusion": c_p},
            },
            "roc": {
                "baseline": roc_b,
                "proposed": roc_p,
            },
            "region_attribution_avg_percent": {
                "baseline": region_avg_b,
                "proposed": region_avg_p,
            },
            "examples": examples,
        }

    report["scenarios"]["normal"] = _scenario_run("normal", lowlight=False)
    report["scenarios"]["lowlight"] = _scenario_run("lowlight", lowlight=True)

    _status_update(
        {
            "state": "running",
            "stage": "writing",
            "message": "Writing report JSON...",
            "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
            "overall_embeddings_total": int(overall_embeddings_total),
            "overall_saliency_done": int(overall_saliency_done_ref[0]),
            "overall_saliency_total": int(overall_saliency_total),
        }
    )

    report = _json_sanitize(report)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(out_json)

    _status_update(
        {
            "state": "done",
            "stage": "done",
            "message": "Done",
            "report": _safe_rel(out_json),
            "overall_embeddings_done": int(overall_embeddings_done_ref[0]),
            "overall_embeddings_total": int(overall_embeddings_total),
            "overall_saliency_done": int(overall_saliency_done_ref[0]),
            "overall_saliency_total": int(overall_saliency_total),
        }
    )

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-dir", default=str(ROOT / "Preprocessed"), help="Path to Preprocessed dataset root")
    ap.add_argument("--pairs-csv", default="", help="Optional CSV of pairs with columns img1_path,img2_path,label")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Base folder for CSV image paths (default: data/)")
    ap.add_argument("--baseline-model", default=str(ROOT / "model" / "Baseline" / "best_model.h5"))
    ap.add_argument("--proposed-model", default=str(ROOT / "model" / "final_best_model.keras"))
    ap.add_argument("--yolov8-weights", default=str(ROOT / "yolov8n.pt"))
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples-per-type", type=int, default=1)
    ap.add_argument("--out-json", default=str(CACHE_DIR / "report.json"))
    ap.add_argument("--heatmap-prefix", default="", help="Subfolder under cache/Parameters/heatmaps (e.g. 'validation')")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--region-sample-n", type=int, default=30, help="How many random pairs to sample for avg region attribution per scenario")
    ap.add_argument("--saliency-batch", type=int, default=8, help="Batch size for gradient saliency computations (uses more RAM/VRAM but faster)")
    ap.add_argument("--tf-intra-threads", type=int, default=0, help="Optional TF intra-op threads (0=leave default)")
    ap.add_argument("--tf-inter-threads", type=int, default=0, help="Optional TF inter-op threads (0=leave default)")
    ap.add_argument("--io-workers", type=int, default=0, help="Parallel workers for image loading/resizing during embedding (0/1=off)")
    ap.add_argument("--emb-cache-dir", default=str(CACHE_DIR / "embeddings_cache"), help="Directory for persistent sqlite embedding caches (empty to disable)")
    ap.add_argument("--proposed-mnetv2-weight", type=float, default=0.4, help="Weight for MobileNetV2 branch in Proposed ensemble (0..1)")
    ap.add_argument("--proposed-optimize", default="f1", help="Threshold objective for Proposed: f1|accuracy|precision|recall|f_beta")
    ap.add_argument("--proposed-beta", type=float, default=1.0, help="Beta for f_beta objective (beta>1 favors recall -> fewer FN)")
    ap.add_argument("--status-json", default="", help="Optional path to write progress status JSON")
    ap.add_argument("--debug", action="store_true", help="Include traceback in output on error")
    args = ap.parse_args()

    pairs_csv = Path(args.pairs_csv) if str(args.pairs_csv).strip() else None
    status_json = Path(args.status_json) if str(args.status_json).strip() else None
    emb_cache_dir = Path(args.emb_cache_dir) if str(args.emb_cache_dir).strip() else None
    try:
        report = run_report(
            pairs_csv=pairs_csv,
            data_root=Path(args.data_root),
            preprocessed_dir=Path(args.preprocessed_dir),
            baseline_model=Path(args.baseline_model),
            proposed_model=Path(args.proposed_model),
            yolov8_weights=Path(args.yolov8_weights),
            max_pairs=max(0, int(args.max_pairs)),
            seed=int(args.seed),
            examples_per_type=max(0, int(args.examples_per_type)),
            heatmap_prefix=str(args.heatmap_prefix),
            batch_size=max(1, int(args.batch_size)),
            region_sample_n=max(0, int(args.region_sample_n)),
            saliency_batch=max(1, int(args.saliency_batch)),
            tf_intra_threads=int(args.tf_intra_threads),
            tf_inter_threads=int(args.tf_inter_threads),
            io_workers=max(0, int(args.io_workers)),
            emb_cache_dir=emb_cache_dir,
            status_json=status_json,
            out_json=Path(args.out_json),
            proposed_mnetv2_weight=float(args.proposed_mnetv2_weight),
            proposed_threshold_optimize=str(args.proposed_optimize),
            proposed_threshold_beta=float(args.proposed_beta),
        )
        print(json.dumps({"ok": True, "report": _json_sanitize(report)}, ensure_ascii=False, allow_nan=False))
    except Exception as e:
        # Best-effort status update on error
        if status_json is not None:
            try:
                StatusWriter(status_json).write({"state": "error", "stage": "error", "error": str(e), "percent": 0})
            except Exception:
                pass
        payload = {"ok": False, "error": str(e)}
        if args.debug:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(_json_sanitize(payload), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
