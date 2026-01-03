#!/usr/bin/env python3
"""
===============================================================================
FURGETMENOT
===============================================================================

This script implements a pet image matching system using a Siamese Capsule
Network with MobileNetV2 as the base model architecture. It finds visually
similar pet images from a preprocessed gallery.

Architecture:
    - Base Model: MobileNetV2 (pretrained)
    - Custom Layers: Enhanced Capsule Layers with attention mechanism
    - Similarity Metric: Cosine similarity between embedding vectors
    
Author: GROUP 2
Date: October 2025
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import sys
import json
import os
import base64
from io import BytesIO
import os
import json
import time
import base64
import hashlib
import threading
import atexit
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Third-party imports - Core scientific computing
import numpy as np                          # Numerical operations and array handling
from PIL import Image                       # Image loading and thumbnail generation

# Third-party imports - Deep learning framework
import tensorflow as tf                     # Core deep learning framework
from tensorflow.keras import backend as K  # Keras backend operations
from tensorflow.keras.models import load_model, Model  # Model loading and creation
from tensorflow.keras.preprocessing import image       # Image preprocessing utilities
from tensorflow.keras.layers import (       # Neural network layers
    Layer, 
    Conv2D, 
    MultiHeadAttention, 
    LayerNormalization
)
from tensorflow.keras.applications import MobileNetV2  # Base model architecture
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Configure warnings and TensorFlow
warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# Optimize TensorFlow to use all available CPU cores for better performance
tf.config.threading.set_intra_op_parallelism_threads(cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(cpu_count())

# ============================================================================
# CACHE MANAGER (Embeddings + Thumbnails)
# ============================================================================

class _JSONCache:
    """
    Simple thread-safe JSON cache loader/saver with lazy writes.
    """
    def __init__(self, path: Path, root_key: str):
        self.path = Path(path)
        self.root_key = root_key
        self._lock = threading.RLock()
        self._dirty = False
        self._data = {
            "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            self.root_key: {}
        }
        try:
            if self.path.exists():
                with self.path.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and self.root_key in loaded:
                        self._data = loaded
        except Exception:
            # Start fresh on read failure
            pass

    def get(self, key):
        with self._lock:
            return self._data.get(self.root_key, {}).get(key)

    def set(self, key, value):
        with self._lock:
            self._data.setdefault(self.root_key, {})[key] = value
            self._dirty = True

    def flush(self):
        with self._lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False)
            tmp.replace(self.path)
            self._dirty = False


class CacheManager:
    """
    Coordinates caches for embeddings and thumbnails.
    - Keys are content-based with path+mtime+size, plus model/config namespace for embeddings.
    """
    def __init__(self):
        # Ensure absolute path to project root (fin)
        root = Path(__file__).resolve().parent.parent  # go up from python/ to fin/
        self.cache_dir = root / 'cache' / 'Proposed'
        self.thumbs_dir = self.cache_dir / 'thumbs'
        self.emb_path = self.cache_dir / 'embeddings_cache.json'
        self.thm_path = self.cache_dir / 'thumbnails_cache.json'
        self.emb_cache = _JSONCache(self.emb_path, 'embeddings')
        self.thm_cache = _JSONCache(self.thm_path, 'thumbnails')
        atexit.register(self.flush)

    @staticmethod
    def _file_fingerprint(path: Path) -> str:
        try:
            stat = path.stat()
            payload = f"{path.resolve()}|{int(stat.st_mtime)}|{stat.st_size}".encode('utf-8', 'ignore')
            return hashlib.sha1(payload).hexdigest()
        except Exception:
            # Fallback to path-only key
            return hashlib.sha1(str(path).encode('utf-8', 'ignore')).hexdigest()

    @staticmethod
    def _model_namespace(engine) -> str:
        model_path = getattr(engine, 'model_path', None)
        use_m = getattr(engine, 'use_mnetv2', True)
        weight = getattr(engine, 'mnetv2_weight', 1.0)
        if model_path:
            p = Path(model_path)
            try:
                stat = p.stat()
                model_fp = f"{p.name}|{int(stat.st_mtime)}|{stat.st_size}"
            except Exception:
                model_fp = p.name
        else:
            model_fp = 'model'
        return f"{model_fp}|use_m={use_m}|w={weight}"

    def embedding_key(self, engine, img_path: str) -> str:
        p = Path(img_path)
        return f"emb|{self._model_namespace(engine)}|{self._file_fingerprint(p)}"

    def thumbnail_key(self, img_path: str, size) -> str:
        p = Path(img_path)
        return f"thm|{size[0]}x{size[1]}|{self._file_fingerprint(p)}"

    def get_embedding(self, key: str):
        return self.emb_cache.get(key)

    def set_embedding(self, key: str, vec: np.ndarray):
        try:
            self.emb_cache.set(key, {
                'updated': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'vector': vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            })
        except Exception:
            # Ignore cache write errors
            pass

    def get_thumbnail(self, key: str):
        return self.thm_cache.get(key)

    def _thumb_path_for_key(self, key: str, ext: str = '.jpg') -> Path:
        """Deterministic thumbnail file path for a given cache key."""
        # Use SHA1(key) to avoid filesystem issues; shard into subdir for scalability
        h = hashlib.sha1(key.encode('utf-8', 'ignore')).hexdigest()
        subdir = h[:2]
        return self.thumbs_dir / subdir / f"{h}{ext}"

    def set_thumbnail_file(self, key: str, data_bytes: bytes, mime: str = 'image/jpeg'):
        """Persist thumbnail to a file and record small JSON entry with file path."""
        try:
            # Ensure directory exists
            self.thumbs_dir.mkdir(parents=True, exist_ok=True)
            dst = self._thumb_path_for_key(key, '.jpg')
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'wb') as f:
                f.write(data_bytes)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # On some environments fsync may not be necessary/available
                    pass
            # Store relative path to keep JSON stable across machines
            rel = os.path.relpath(dst, self.cache_dir)
            self.thm_cache.set(key, {
                'updated': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                'mime': mime,
                'file': rel.replace('\\', '/')
            })
        except Exception:
            # Swallow cache errors
            pass

    def set_thumbnail(self, key: str, b64: str, mime: str = 'image/jpeg'):
        """Backward-compatible: accept base64 string and persist to file."""
        try:
            data_bytes = base64.b64decode(b64.encode('utf-8'))
            self.set_thumbnail_file(key, data_bytes, mime=mime)
        except Exception:
            pass

    def flush(self):
        # Persist to disk
        try:
            self.emb_cache.flush()
            self.thm_cache.flush()
        except Exception:
            pass


# Singleton cache manager
_CACHE = CacheManager()
# Eagerly ensure directory exists
try:
    _CACHE.cache_dir.mkdir(parents=True, exist_ok=True)
    _CACHE.thumbs_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# ============================================================================
# CAPSULE NETWORK UTILITY FUNCTIONS
# ============================================================================

def squash(vectors, axis=-1):
    """
    Squashing activation function for capsule networks.
    
    Maps input vectors to output vectors with lengths between 0 and 1,
    preserving direction while scaling magnitude based on vector norm.
    This ensures short vectors get shrunk to almost zero length and long
    vectors get shrunk to a length slightly below 1.
    
    Args:
        vectors: Input tensor of capsule vectors
        axis: Axis along which to compute the norm (default: -1)
    
    Returns:
        Squashed vectors with improved numerical stability
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    epsilon = 1e-7  # Small constant for numerical stability
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + epsilon)
    return scale * vectors


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    """
    Calculate vector norm with numerical stability.
    
    Computes the L2 norm while avoiding division by zero and numerical
    underflow issues that can occur with very small values.
    
    Args:
        s: Input tensor
        axis: Axis along which to compute norm
        epsilon: Small constant to prevent division by zero
        keep_dims: Whether to keep reduced dimensions
        name: Optional name for the operation
    
    Returns:
        Safe L2 norm of the input tensor
    """
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keep_dims)
    return tf.sqrt(squared_norm + epsilon)


# ============================================================================
# CUSTOM CAPSULE LAYER IMPLEMENTATIONS
# ============================================================================

class CapsuleLayer(Layer):
    """
    Base Capsule Layer implementation.
    
    A capsule is a group of neurons whose activity vector represents the
    instantiation parameters of a specific type of entity. This base class
    provides the foundation for capsule operations.
    
    Args:
        num_capsules: Number of capsule units in this layer
        dim_capsules: Dimensionality of each capsule's output vector
        routings: Number of routing iterations (default: 3)
    """
    
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
    
    def build(self, input_shape):
        super(CapsuleLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        return inputs
    
    def get_config(self):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsules': self.dim_capsules,
            'routings': self.routings
        })
        return config


class PrimaryCapsule(Layer):
    """
    Primary Capsule Layer - First capsule layer in the network.
    
    Converts convolutional feature maps into capsule format by applying
    convolution followed by reshaping and squashing. Acts as a bridge
    between traditional CNN layers and capsule layers.
    
    Args:
        dim_capsules: Dimensionality of each capsule vector
        n_channels: Number of capsule channels
        kernel_size: Size of convolutional kernel
        strides: Stride of convolution
        padding: Padding mode ('valid' or 'same')
    """
    
    def __init__(self, dim_capsules, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsules = dim_capsules
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
    def build(self, input_shape):
        self.conv = Conv2D(
            filters=self.dim_capsules * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation='relu',
            name=f'{self.name}_conv'
        )
        self.conv.build(input_shape)
        super(PrimaryCapsule, self).build(input_shape)
        
    def call(self, inputs, training=None):
        # Apply convolution
        outputs = self.conv(inputs)
        
        # Reshape to capsule format: [batch, num_capsules, dim_capsules]
        batch_size = tf.shape(outputs)[0]
        outputs = tf.reshape(outputs, [batch_size, -1, self.dim_capsules])
        
        # Apply squash activation
        return squash(outputs, axis=-1)
    
    def get_config(self):
        config = super(PrimaryCapsule, self).get_config()
        config.update({
            'dim_capsules': self.dim_capsules,
            'n_channels': self.n_channels,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class EnhancedCapsuleLayer(CapsuleLayer):
    """
    Enhanced Capsule Layer with Self-Attention and Dynamic Routing.
    
    Implements an advanced capsule layer that combines:
    1. Dynamic routing by agreement - Routes information based on agreement
    2. Self-attention mechanism - Captures long-range dependencies
    3. Layer normalization - Stabilizes training
    4. Routing entropy monitoring - Tracks routing confidence
    
    This layer is the core of the pet matching system, enabling the model
    to learn rich, part-whole relationships in pet images.
    
    Args:
        num_capsules: Number of output capsules
        dim_capsules: Dimensionality of each output capsule
        routings: Number of dynamic routing iterations
        attention_heads: Number of attention heads for multi-head attention
        use_attention: Whether to use self-attention mechanism
        kernel_initializer: Initializer for weight matrices
    """
    
    def __init__(self, num_capsules, dim_capsules, routings=3, attention_heads=4, 
                 use_attention=True, kernel_initializer='glorot_uniform', **kwargs):
        super(EnhancedCapsuleLayer, self).__init__(
            num_capsules, dim_capsules, routings, **kwargs
        )
        self.attention_heads = attention_heads
        self.use_attention = use_attention
        self.kernel_initializer = kernel_initializer
        
    def build(self, input_shape):
        super().build(input_shape)
        self.dim_in = int(input_shape[-1])
        
        # Weight matrix for transforming input capsules to output capsules
        self.W = self.add_weight(
            shape=(self.num_capsules, self.dim_in, self.dim_capsules),
            initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_avg', distribution='uniform'
            ),
            name=f"{self.name}_W"
        )
        
        # Bias for each output capsule
        self.b_caps = self.add_weight(
            shape=(self.num_capsules, self.dim_capsules),
            initializer='zeros',
            name=f"{self.name}_b"
        )
        
        # Scaling factor for output
        self.gamma = self.add_weight(
            shape=(1,), 
            initializer=tf.keras.initializers.constant(3.0), 
            name=f"{self.name}_gamma"
        )
        
        # Track routing entropy for monitoring
        self.last_c_entropy = None
        
        # Self-attention mechanism for capturing global context
        if self.use_attention:
            unique_prefix = f"{self.name}_"
            self.attention_layer = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.dim_capsules,
                name=f'{unique_prefix}capsule_attention'
            )
            self.attention_norm = LayerNormalization(
                name=f'{unique_prefix}attention_norm'
            )
    
    def enhanced_dynamic_routing(self, u_hat, training):
        """
        Dynamic routing by agreement algorithm.
        
        Iteratively updates routing coefficients based on agreement between
        lower-level and higher-level capsules. This allows the network to
        learn part-whole relationships automatically.
        
        Args:
            u_hat: Predicted output vectors from lower capsules
            training: Whether in training mode
        
        Returns:
            Routed output capsules
        """
        batch_size = tf.shape(u_hat)[0]
        input_num_capsules = tf.shape(u_hat)[1]
        num_output_capsules = tf.shape(u_hat)[2] 
        capsule_dim = tf.shape(u_hat)[3]
        
        # Initialize routing logits
        b = tf.zeros([batch_size, input_num_capsules, num_output_capsules, 1])
        
        # Dynamic routing iterations
        for i in range(self.routings):
            # Compute routing coefficients using softmax
            c = tf.nn.softmax(b, axis=2)
            
            # Weighted sum of predictions
            s = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)
            
            # Squash to get output capsule
            v = squash(s, axis=-1)
            
            # Update routing logits based on agreement (except last iteration)
            if i < self.routings - 1:
                agreement = tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)
                b = b + agreement
        
        # Calculate routing entropy for monitoring
        c_final = tf.nn.softmax(b, axis=2)
        entropy = -tf.reduce_sum(c_final * tf.math.log(c_final + 1e-9), axis=2)
        norm_entropy = entropy / tf.math.log(tf.cast(num_output_capsules, tf.float32) + 1e-9)
        self.last_c_entropy = tf.reduce_mean(norm_entropy)
        
        return tf.squeeze(v, axis=1)
    
    def call(self, inputs, training=None):
        """Forward pass through the enhanced capsule layer."""
        # Transform input capsules to output space
        u_hat = tf.einsum('b n d, c d h -> b n c h', inputs, self.W) + self.bias_expand()
        
        # Add noise during training for regularization
        if training:
            u_hat += tf.random.normal(tf.shape(u_hat), stddev=0.05)
            
            # Add extra jitter if predictions are too certain
            global_std = tf.math.reduce_std(u_hat)
            def add_jitter():
                return u_hat + tf.random.normal(tf.shape(u_hat), stddev=0.1)
            u_hat = tf.cond(global_std < 0.02, add_jitter, lambda: u_hat)
        
        # Apply dynamic routing
        routed = self.enhanced_dynamic_routing(u_hat, training=training if training is not None else False)
        
        # Apply self-attention if enabled
        if self.use_attention:
            attended = self.attention_layer(routed, routed, training=training)
            routed = self.attention_norm(attended + routed, training=training)
        
        # Apply squash and scaling
        routed = squash(routed, axis=-1) * self.gamma
        return routed

    def bias_expand(self):
        """Expand bias to match batch dimension."""
        return tf.reshape(self.b_caps, (1, 1, self.num_capsules, self.dim_capsules))

    def get_last_routing_entropy(self):
        """Get the last computed routing entropy for monitoring."""
        return self.last_c_entropy
    
    def get_config(self):
        config = super(EnhancedCapsuleLayer, self).get_config()
        config.update({
            'attention_heads': self.attention_heads,
            'use_attention': self.use_attention
        })
        return config


# ============================================================================
# SIMILARITY METRIC FUNCTIONS
# ============================================================================

def cosine_distance(vectors):
    """
    Compute cosine distance between embedding vector pairs.
    
    Cosine distance measures the angular difference between vectors,
    making it invariant to magnitude. This is ideal for comparing
    embeddings where we care about semantic similarity rather than
    absolute values.
    
    Formula: distance = 1 - cosine_similarity
    where cosine_similarity = (x · y) / (||x|| * ||y||)
    
    Args:
        vectors: Tuple of (x, y) where x and y are embedding vectors
    
    Returns:
        Cosine distance in range [0, 1], where:
            0 = identical direction (most similar)
            1 = opposite direction (most dissimilar)
    """
    x, y = vectors
    
    # Normalize vectors to unit length
    x_norm = K.l2_normalize(x, axis=1)
    y_norm = K.l2_normalize(y, axis=1)
    
    # Compute cosine similarity
    cosine_sim = K.sum(x_norm * y_norm, axis=1)
    
    # Convert to distance and clip to valid range
    cosine_dist = 1.0 - cosine_sim
    return K.clip(cosine_dist, 0.0, 1.0)


# ============================================================================
# LOSS AND METRIC FUNCTIONS
# ============================================================================

def loss_fn(y_true, y_pred):
    """
    Enhanced contrastive loss function for Siamese network training.
    
    This loss function encourages similar pairs (same pet) to have small
    distances and dissimilar pairs (different pets) to have large distances
    beyond a margin. Includes several enhancements:
    
    1. Basic contrastive loss with asymmetric weighting
    2. Hard negative mining to focus on difficult examples
    3. Separation regularization to push embeddings apart
    4. Focal loss weighting to handle class imbalance
    
    Args:
        y_true: Ground truth labels (1 = similar, 0 = dissimilar)
        y_pred: Predicted distances from the model
    
    Returns:
        Combined loss value for optimization
    """
    # Ensure consistent data types
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Reshape for consistent dimensions
    batch = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch, 1))
    y_pred = tf.reshape(y_pred, (batch, -1))
    y_pred = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    
    # Loss hyperparameters
    margin = 0.5              # Minimum distance for dissimilar pairs
    label_smoothing = 0.0     # Label smoothing factor (disabled)
    alpha = 0.1               # Weight for separation regularization
    
    # Apply label smoothing (currently disabled)
    y_true_smooth = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    margin_tf = tf.constant(margin, dtype=tf.float32)
    
    # Asymmetric weighting: penalize false negatives more
    pos_weight = 1.0   # Weight for similar pairs
    neg_weight = 1.35  # Weight for dissimilar pairs (higher to reduce false negatives)
    
    # Contrastive loss components
    pos_loss = pos_weight * y_true_smooth * K.square(y_pred)  # Penalize large distances for similar pairs
    neg_loss = neg_weight * (1.0 - y_true_smooth) * K.square(K.maximum(margin_tf - y_pred, 0.0))  # Penalize small distances for dissimilar pairs
    contrastive = K.mean(pos_loss + neg_loss)
    
    # Hard negative mining: focus on top 10% most difficult negative examples
    k = tf.maximum(1, tf.cast(0.1 * tf.cast(batch, tf.float32), tf.int32))
    neg_residual = K.maximum(margin_tf - y_pred, 0.0) * (1.0 - y_true_smooth)
    topk_vals, _ = tf.math.top_k(tf.reshape(neg_residual, (-1,)), k=k)
    hard_neg_loss = K.mean(K.square(topk_vals))
    
    # Separation regularization: encourage decision boundary at margin/2
    sep_reg = K.mean(K.exp(-5.0 * K.square(y_pred - margin_tf * 0.5)))
    
    # Focal loss weighting: focus on hard-to-classify examples
    focal_weight = K.square(K.abs(y_true_smooth - K.sigmoid(1.0 - y_pred)))
    focal_loss = K.mean(focal_weight * (pos_loss + neg_loss))
    
    # Combine all loss components
    return contrastive + 0.3 * hard_neg_loss + alpha * sep_reg + 0.1 * focal_loss


def accuracy_metric(y_true, y_pred):
    """
    Accuracy metric for distance predictions.
    
    Measures the percentage of pairs correctly classified as similar or
    dissimilar based on a distance threshold.
    
    Args:
        y_true: Ground truth labels (1 = similar, 0 = dissimilar)
        y_pred: Predicted distances
    
    Returns:
        Accuracy in range [0, 1]
    """
    threshold = 0.5  # Distance threshold for classification
    
    # Convert distances to binary predictions
    y_pred_normalized = y_pred
    predictions = K.cast(y_pred_normalized < threshold, K.floatx())
    
    # Compare with ground truth
    y_true_cast = K.cast(y_true, K.floatx())
    correct = K.cast(K.equal(y_true_cast, predictions), K.floatx())
    accuracy = K.mean(correct)
    
    # Handle edge cases (NaN values)
    accuracy = tf.where(tf.math.is_finite(accuracy), accuracy, 0.5)
    return K.clip(accuracy, 0.0, 1.0)

# ============================================================================
# PET MATCHING ENGINE
# ============================================================================

class PetMatchingEngine:
    """
    Pet Image Matching Engine using Siamese Network with MobileNetV2 Base.
    
    This engine performs similarity-based pet image matching by:
    1. Loading a trained Siamese Capsule Network model with MobileNetV2 base
    2. Extracting deep embedding vectors from pet images
    3. Computing cosine similarities between embeddings
    4. Ranking and returning the most similar matches
    
    The system uses an ensemble approach combining:
    - Custom capsule network features (learned pet-specific patterns)
    - MobileNetV2 base model features (general visual understanding)
    
    Architecture:
        Query Image -> MobileNetV2 Base -> Capsule Layers -> Embedding
                                                               ↓
                                                         Cosine Similarity
                                                               ↓
        Gallery Images -> [Same Pipeline] -----------------> Top-K Matches
    
    Args:
        model_path: Path to the trained .keras model file
        debug: Enable debug logging (default: False)
        batch_size: Number of images to process in parallel (default: 32)
        num_workers: Number of worker threads (default: CPU count)
        use_mnetv2: Use MobileNetV2 base model ensemble (default: True)
        mnetv2_weight: Weight for MobileNetV2 features, 0.0-1.0 (default: 1.0)
                       Higher values give more weight to base model features
    
    Example:
        engine = PetMatchingEngine('model.keras', debug=True)
        engine.load_model()
        engine.load_mnetv2_model()
        matches = engine.find_matches('query.jpg', 'gallery/', 'cat,dog', top_k=10)
    """
    
    def __init__(self, model_path, debug=False, batch_size=32, num_workers=None, 
                 use_mnetv2=True, mnetv2_weight=1.0):
        """Initialize the matching engine with configuration parameters."""
        self.model_path = model_path
        self.debug = debug
        
        # Model components
        self.model = None              # Full Siamese model
        self.embedding_model = None     # Base network for embeddings
        self.mnetv2_model = None       # MobileNetV2 base model
        
        # Ensemble configuration
        self.use_mnetv2 = use_mnetv2
        self.mnetv2_weight = mnetv2_weight
        
        # Performance configuration
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers else cpu_count()
        
        # Debug tracking
        self.debug_info = []
        
        # Log initialization
        self.log(f"Initialized with {self.num_workers} workers and batch size {self.batch_size}")
        if self.use_mnetv2:
            self.log(f"MobileNetV2 base model enabled with weight: {self.mnetv2_weight}")
        
    def log(self, msg):
        """
        Log debug messages to stderr and internal list.
        
        Args:
            msg: Message string to log
        """
        if self.debug:
            self.debug_info.append(msg)
            print(f"[DEBUG] {msg}", file=sys.stderr)
    
    def load_model(self):
        """
        Load the trained Siamese Capsule Network model.
        
        This method:
        1. Loads the full Siamese model from disk with custom layers
        2. Extracts the base embedding network for inference
        3. Prepares the model for feature extraction
        
        The model architecture includes:
        - MobileNetV2 base (feature extraction)
        - Primary Capsule layers (low-level part detection)
        - Enhanced Capsule layers (high-level pattern recognition)
        - Attention mechanisms (global context integration)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        self.log(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Define custom objects required for model loading
        # These functions and classes are used in the saved model architecture
        custom_objects = {
            'squash': squash,
            'safe_norm': safe_norm,
            'CapsuleLayer': CapsuleLayer,
            'PrimaryCapsule': PrimaryCapsule,
            'EnhancedCapsuleLayer': EnhancedCapsuleLayer,
            'cosine_distance': cosine_distance,
            'loss_fn': loss_fn,
            'accuracy_metric': accuracy_metric
        }
        
        # Try loading the main model with fallback options
        model_loaded = False
        models_to_try = [
            self.model_path,  # Main model
            self.model_path.replace('.keras', '-1.keras'),  # Backup 1
            self.model_path.replace('.keras', '-2.keras'),  # Backup 2
            self.model_path.replace('.keras', '-3.keras'),  # Backup 3
        ]
        
        for model_file in models_to_try:
            if not os.path.exists(model_file):
                continue
                
            try:
                self.log(f"Attempting to load: {os.path.basename(model_file)}")
                
                # Load model with custom objects
                # safe_mode=False: Allows Lambda layers in the architecture
                # compile=False: Skip metric compilation (not needed for inference)
                self.model = load_model(
                    model_file, 
                    custom_objects=custom_objects, 
                    safe_mode=False, 
                    compile=False
                )
                self.log(f"✓ Model loaded successfully from {os.path.basename(model_file)}")
                self.log(f"  Model inputs: {len(self.model.inputs)}, outputs: {len(self.model.outputs)}")
                
                # Extract the base network for embedding extraction
                # The siamese model has 2 inputs (anchor, positive) and 1 output (distance)
                # We need to access the base network that produces embeddings
                if len(self.model.layers) > 2:
                    # Find the base network layer (MobileNetV2_CapsNet_Hybrid)
                    for layer in self.model.layers:
                        if 'MobileNetV2' in layer.name or 'Hybrid' in layer.name:
                            self.embedding_model = layer
                            self.log(f"  Found embedding model: {layer.name}")
                            break
                
                if self.embedding_model is None:
                    self.log("  Using full model for embeddings")
                    # If we can't find the base network, create a wrapper
                    # that uses just the first input
                    try:
                        self.embedding_model = Model(
                            inputs=self.model.inputs[0],
                            outputs=self.model.layers[2].output[1]  # Get embeddings from base network
                        )
                    except:
                        # If that fails, we'll rely on MobileNetV2 only
                        self.log("  Warning: Could not extract embedding model, will use MobileNetV2 only")
                        pass
                
                model_loaded = True
                break  # Success! Exit the loop
                
            except Exception as e:
                self.log(f"✗ Failed to load {os.path.basename(model_file)}: {str(e)}")
                continue  # Try next model
        
        if not model_loaded:
            self.log("=" * 60)
            self.log("WARNING: All custom models failed to load!")
            self.log("Falling back to MobileNetV2-only mode (reduced accuracy)")
            self.log("=" * 60)
            # We'll continue with just MobileNetV2 (loaded separately below)
            # This provides basic functionality but with reduced matching accuracy
            self.model = None
            self.embedding_model = None
    
    def load_mnetv2_model(self):
        """
        Load MobileNetV2 base model for ensemble matching.
        
        MobileNetV2 is a lightweight convolutional neural network architecture
        designed for mobile and embedded vision applications. It serves as the
        base feature extractor in our matching system.
        
        Features:
        - Pretrained on ImageNet (1000 classes, 1.2M images)
        - Efficient inverted residual structure
        - Linear bottlenecks for better feature preservation
        - Global average pooling for 1280-dimensional embeddings
        
        The model is used alongside the custom capsule network to provide:
        1. General visual understanding (shapes, textures, colors)
        2. Transfer learning benefits from large-scale pretraining
        3. Robust baseline features for similarity comparison
        
        Raises:
            Exception: If model loading fails
        """
        self.log("Loading MobileNetV2 base model...")
        
        try:
            # Load MobileNetV2 with pretrained weights
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'  # Global average pooling
            )
            
            # Create embedding model
            self.mnetv2_model = Model(
                inputs=base_model.input,
                outputs=base_model.output
            )
            
            self.log(f"MobileNetV2 model loaded successfully. Output shape: {self.mnetv2_model.output_shape}")
            
        except Exception as e:
            self.log(f"Error loading MobileNetV2 model: {str(e)}")
            self.use_mnetv2 = False
            raise
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        """
        Load and preprocess an image for the custom capsule network.
        
        Preprocessing steps:
        1. Load image from disk
        2. Resize to target dimensions
        3. Convert to numpy array
        4. Normalize pixel values to [0, 1] range
        5. Add batch dimension
        
        Args:
            img_path: Path to the image file
            target_size: Target dimensions (height, width), default: (224, 224)
        
        Returns:
            Preprocessed image array with shape (1, 224, 224, 3)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        self.log(f"Preprocessing image: {img_path}")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load and resize image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        
        # Normalize to [0, 1] range for capsule network
        img_array = img_array / 255.0
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_image_for_mnetv2(self, img_path, target_size=(224, 224)):
        """
        Load and preprocess an image for MobileNetV2 base model.
        
        Uses MobileNetV2-specific preprocessing which applies:
        - Scaling to [-1, 1] range (different from capsule network)
        - Channel-wise mean subtraction
        - Optimized for MobileNetV2's training procedure
        
        Args:
            img_path: Path to the image file
            target_size: Target dimensions (height, width), default: (224, 224)
        
        Returns:
            Preprocessed image array with shape (1, 224, 224, 3)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load and resize image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply MobileNetV2-specific preprocessing
        img_array = mobilenet_preprocess(img_array)
        
        return img_array
    
    def get_embedding(self, img_path):
        """
        Extract feature embedding from an image using the custom capsule network.
        
        The capsule network produces a high-dimensional embedding that captures
        hierarchical features and spatial relationships between parts of the image.
        
        Args:
            img_path: Path to the image file
        
        Returns:
            1D numpy array containing the flattened embedding vector
        
        Note:
            - Model may output [capsules, embeddings] - we use embeddings (second element)
            - Output is automatically flattened to 1D vector for similarity computation
            - Falls back to MobileNetV2 if custom model is not available
        """
        # If embedding model is not available, fall back to MobileNetV2
        if self.embedding_model is None:
            return self.get_mnetv2_embedding(img_path)
        
        img_array = self.preprocess_image(img_path)
        
        # Get embeddings
        output = self.embedding_model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if isinstance(output, list):
            # If output is [capsules, embeddings], take embeddings (second element)
            embedding = output[1]
        else:
            embedding = output
        
        # Flatten to 1D vector
        embedding = embedding.flatten()
        
        return embedding
    
    def get_mnetv2_embedding(self, img_path):
        """
        Extract feature embedding from an image using MobileNetV2 base model.
        
        MobileNetV2 provides general-purpose visual features pretrained on ImageNet,
        offering robust feature extraction across diverse image types.
        
        Args:
            img_path: Path to the image file
        
        Returns:
            1D numpy array (1280 dimensions) containing the embedding vector,
            or None if MobileNetV2 is not enabled
        """
        if not self.use_mnetv2 or self.mnetv2_model is None:
            return None
        
        img_array = self.preprocess_image_for_mnetv2(img_path)
        
        # Get MobileNetV2 embeddings
        embedding = self.mnetv2_model.predict(img_array, verbose=0)
        
        # Flatten to 1D vector
        embedding = embedding.flatten()
        
        return embedding
    
    def get_combined_embedding(self, img_path):
        """
        Extract combined embedding using ensemble of capsule network and MobileNetV2.
        
        Ensemble Strategy:
        1. Extract embeddings from both models
        2. L2-normalize each embedding vector
        3. Weight by mnetv2_weight parameter
        4. Concatenate weighted embeddings
        5. Re-normalize final embedding
        
        This approach preserves information from both models while allowing
        tuning of their relative importance via mnetv2_weight.
        
        Args:
            img_path: Path to the image file
        
        Returns:
            1D numpy array containing the combined embedding vector
        
        Note:
            If MobileNetV2 is disabled, returns only capsule network embedding
        """
        # Cache first
        try:
            cache_key = _CACHE.embedding_key(self, img_path)
            cached = _CACHE.get_embedding(cache_key)
            if cached and 'vector' in cached:
                vec = np.array(cached['vector'], dtype=np.float32)
                if vec.size > 0:
                    return vec
        except Exception:
            pass

        # Get custom model embedding (may fall back to MobileNetV2 if custom model unavailable)
        custom_emb = self.get_embedding(img_path)
        
        # If custom model is unavailable, we're already using MobileNetV2
        if self.embedding_model is None:
            try:
                _CACHE.set_embedding(cache_key, np.array(custom_emb))
            except Exception:
                pass
            return custom_emb
        
        if not self.use_mnetv2:
            try:
                _CACHE.set_embedding(cache_key, np.array(custom_emb))
            except Exception:
                pass
            return custom_emb
        
        # Get MobileNetV2 embedding
        mnetv2_emb = self.get_mnetv2_embedding(img_path)
        
        if mnetv2_emb is None:
            return custom_emb
        
        # Handle dimension mismatch by concatenating instead of weighted average
        # Normalize both embeddings first
        custom_emb_norm = custom_emb / (np.linalg.norm(custom_emb) + 1e-10)
        mnetv2_emb_norm = mnetv2_emb / (np.linalg.norm(mnetv2_emb) + 1e-10)
        
        # Concatenate embeddings (this preserves information from both)
        combined = np.concatenate([
            custom_emb_norm * (1.0 - self.mnetv2_weight),
            mnetv2_emb_norm * self.mnetv2_weight
        ])
        
        # Re-normalize the combined embedding
        combined = combined / (np.linalg.norm(combined) + 1e-10)
        
        try:
            _CACHE.set_embedding(cache_key, np.array(combined))
        except Exception:
            pass
        return combined
    
    def get_embeddings_batch(self, img_paths):
        """
        Extract embeddings for multiple images in batch (optimized for performance).
        
        Batch processing is significantly faster than sequential processing because:
        1. GPU/CPU parallelization for neural network inference
        2. Reduced overhead from model initialization
        3. Memory-efficient vectorized operations
        
        Process:
        1. Preprocess all images (with error handling for corrupt files)
        2. Stack into batch tensor
        3. Run batch prediction on custom model
        4. Run batch prediction on MobileNetV2 (if enabled)
        5. Combine embeddings using ensemble strategy
        
        Args:
            img_paths: List of image file paths to process
        
        Returns:
            Tuple of (embeddings, valid_paths):
            - embeddings: 2D numpy array (N, embedding_dim) where N is number of valid images
            - valid_paths: List of successfully processed image paths
        
        Note:
            Images that fail preprocessing are skipped and logged
        """
        if len(img_paths) == 0:
            return np.array([]), []

        # Prepare outputs, using cache when available
        outputs = [None] * len(img_paths)
        valid_paths = []
        to_process = []  # indices
        uncached_paths = []

        for idx, p in enumerate(img_paths):
            try:
                key = _CACHE.embedding_key(self, p)
                cached = _CACHE.get_embedding(key)
                if cached and 'vector' in cached:
                    vec = np.array(cached['vector'], dtype=np.float32)
                    outputs[idx] = vec
                    valid_paths.append(p)
                else:
                    to_process.append(idx)
                    uncached_paths.append(p)
            except Exception:
                to_process.append(idx)
                uncached_paths.append(p)

        if len(uncached_paths) == 0:
            # All cached; ensure order matches original img_paths
            final_vecs = []
            final_paths = []
            for i, p in enumerate(img_paths):
                if outputs[i] is not None:
                    final_vecs.append(outputs[i])
                    final_paths.append(p)
            return np.vstack(final_vecs), final_paths

        # Preprocess only uncached
        img_arrays = []
        img_arrays_mnetv2 = []
        actual_paths = []  # paths that succeeded preprocessing
        actual_indices = []

        for i, p in zip(to_process, uncached_paths):
            try:
                arr = self.preprocess_image(p)
                img_arrays.append(arr)
                if self.use_mnetv2:
                    arr2 = self.preprocess_image_for_mnetv2(p)
                    img_arrays_mnetv2.append(arr2)
                actual_paths.append(p)
                actual_indices.append(i)
            except Exception as e:
                self.log(f"Error preprocessing {p}: {str(e)}")
                outputs[i] = None

        if len(actual_paths) > 0:
            batch = np.vstack(img_arrays)
            output = self.embedding_model.predict(batch, verbose=0, batch_size=self.batch_size)
            if isinstance(output, list):
                embeddings = output[1]
            else:
                embeddings = output
            embeddings = embeddings.reshape(embeddings.shape[0], -1)

            if self.use_mnetv2 and len(img_arrays_mnetv2) == len(actual_paths):
                batch_mnetv2 = np.vstack(img_arrays_mnetv2)
                mnetv2_embeddings = self.mnetv2_model.predict(batch_mnetv2, verbose=0, batch_size=self.batch_size)
                mnetv2_embeddings = mnetv2_embeddings.reshape(mnetv2_embeddings.shape[0], -1)

                embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
                mnetv2_norm = mnetv2_embeddings / (np.linalg.norm(mnetv2_embeddings, axis=1, keepdims=True) + 1e-10)
                embeddings_weighted = embeddings_norm * (1.0 - self.mnetv2_weight)
                mnetv2_weighted = mnetv2_norm * self.mnetv2_weight
                combined = np.concatenate([embeddings_weighted, mnetv2_weighted], axis=1)
                combined = combined / (np.linalg.norm(combined, axis=1, keepdims=True) + 1e-10)
                computed = combined
            else:
                computed = embeddings

            # Place into outputs and cache
            for j, idx in enumerate(actual_indices):
                vec = computed[j]
                outputs[idx] = vec
                valid_paths.append(actual_paths[j])
                try:
                    key = _CACHE.embedding_key(self, actual_paths[j])
                    _CACHE.set_embedding(key, np.array(vec))
                except Exception:
                    pass

        # Filter out any Nones (failed preprocessing)
        final_vecs = []
        final_paths = []
        for i, p in enumerate(img_paths):
            if outputs[i] is not None:
                final_vecs.append(outputs[i])
                final_paths.append(p)
        if len(final_vecs) == 0:
            return np.array([]), []
        return np.vstack(final_vecs), final_paths
    
    def cosine_similarity(self, vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between vectors,
        ranging from -1 (opposite) to 1 (identical direction).
        
        Process:
        1. L2-normalize both vectors
        2. Compute dot product (cosine of angle)
        3. Map from [-1, 1] to [0, 1] for easier interpretation
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
        
        Returns:
            Similarity score in range [0, 1] where:
            - 0.0 = completely dissimilar (opposite directions)
            - 0.5 = orthogonal (no similarity)
            - 1.0 = identical (same direction)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        similarity = (similarity + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        return float(similarity)
    
    def generate_thumbnail(self, img_path, size=(300, 300)):
        """
        Generate a base64-encoded thumbnail of an image for web display.
        
        Thumbnails are:
        1. Resized to target dimensions while maintaining aspect ratio
        2. Converted to RGB color space for consistency
        3. JPEG-compressed at quality 85 for balance of size/quality
        4. Base64-encoded for embedding in JSON/HTML
        
        Args:
            img_path: Path to the image file
            size: Thumbnail dimensions (width, height), default: (150, 150)
        
        Returns:
            Base64-encoded JPEG string, or None if generation fails
        
        Usage:
            The returned string can be embedded in HTML as:
            <img src="data:image/jpeg;base64,{thumbnail}">
        """
        # Normalize path FIRST for consistent cache keys
        img_path = str(Path(img_path).resolve())
        key = _CACHE.thumbnail_key(img_path, size)
        
        # Try cache first
        try:
            cached = _CACHE.get_thumbnail(key)
            # New format: JSON points to a file path
            if cached and isinstance(cached, dict):
                if 'file' in cached:
                    thumb_file = Path(_CACHE.cache_dir) / cached['file']
                    if thumb_file.exists():
                        self.log(f"Using cached thumbnail for {img_path}")
                        with open(thumb_file, 'rb') as f:
                            return base64.b64encode(f.read()).decode('utf-8')
                # Backward-compat: older cache may still have base64
                if 'data_base64' in cached:
                    return cached['data_base64']
        except Exception as e:
            self.log(f"Cache read error for {img_path}: {str(e)}")

        # Generate new thumbnail
        try:
            if not Path(img_path).exists():
                self.log(f"ERROR: Image file not found: {img_path}")
                return None
                
            img = Image.open(img_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            data = buffer.read()
            # Persist to file cache (JSON stores path only)
            try:
                _CACHE.set_thumbnail_file(key, data, mime='image/jpeg')
                self.log(f"Cached thumbnail for {img_path}: {key}")
            except Exception as e:
                self.log(f"ERROR: Failed to cache thumbnail for {img_path}: {str(e)}")
            # Return base64 for API consumption
            return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            self.log(f"Error generating thumbnail for {img_path}: {str(e)}")
            return None
    
    def generate_thumbnails_parallel(self, img_paths, size=(150, 150)):
        """
        Generate thumbnails for multiple images in parallel using thread pool.
        
        Uses ThreadPoolExecutor to parallelize I/O-bound thumbnail generation.
        Number of workers scales with CPU count for optimal performance.
        
        Args:
            img_paths: List of image file paths
            size: Thumbnail dimensions (width, height), default: (150, 150)
        
        Returns:
            List of base64-encoded thumbnail strings (same order as img_paths)
        
        Note:
            Failed thumbnails return None in their position
        """
        thumbnails = {}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all thumbnail generation tasks
            future_to_path = {executor.submit(self.generate_thumbnail, path, size): path 
                             for path in img_paths}
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    thumb = future.result()
                    thumbnails[path] = thumb
                except Exception as e:
                    self.log(f"Thumbnail generation failed for {path}: {str(e)}")
                    thumbnails[path] = None
        
        return thumbnails

    def _project_embeddings_2d(self, vectors: np.ndarray) -> np.ndarray:
        """Project high-dimensional embeddings into 2D using PCA/SVD."""
        if vectors.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if vectors.shape[0] == 1:
            return np.zeros((1, 2), dtype=np.float32)

        centered = vectors - np.mean(vectors, axis=0, keepdims=True)

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            components = vh[:2]
            coords = np.dot(centered, components.T)
        except np.linalg.LinAlgError:
            if centered.shape[1] >= 2:
                coords = centered[:, :2]
            else:
                coords = np.hstack([
                    centered,
                    np.zeros((centered.shape[0], 2 - centered.shape[1]))
                ])

        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val

        if coords.shape[1] < 2:
            pad = np.zeros((coords.shape[0], 2 - coords.shape[1]))
            coords = np.hstack([coords, pad])

        return coords.astype(np.float32)

    def _resolve_embedding_cap(self, available: int) -> int:
        """Determine how many points to plot; defaults to all available.

        Optional override via EMBEDDING_MAP_MAX_POINTS environment variable.
        """
        if available <= 0:
            return 0
        env_val = os.environ.get('EMBEDDING_MAP_MAX_POINTS')
        if env_val:
            try:
                parsed = int(env_val)
                if parsed > 0:
                    return min(parsed, available)
            except ValueError:
                pass
        return available

    def _build_embedding_map(self, query_embedding, matches, embedding_lookup, max_points=100):
        """Create normalized 2D coordinates for query and top matches."""
        if not matches:
            return None

        selected = []
        for match in matches:
            vec = embedding_lookup.get(match['path'])
            if vec is None:
                continue
            selected.append((match, vec))
            if len(selected) >= max_points:
                break

        if not selected:
            return None

        stacked = np.vstack([query_embedding] + [vec for _, vec in selected]).astype(np.float32)
        coords = self._project_embeddings_2d(stacked)
        if coords.size == 0:
            return None

        points = []
        points.append({
            'role': 'query',
            'label': 'Query Image',
            'type': 'Query',
            'path': None,
            'rank': 0,
            'similarity': 1.0,
            'distance': 0.0,
            'x': round(float(coords[0, 0]), 6),
            'y': round(float(coords[0, 1]), 6)
        })

        for idx, (match, _) in enumerate(selected, start=1):
            coord = coords[idx]
            points.append({
                'role': 'match',
                'label': match['filename'],
                'type': match['type'],
                'path': match['path'],
                'rank': match['rank'],
                'similarity': match['similarity'],
                'distance': match['distance'],
                'x': round(float(coord[0]), 6),
                'y': round(float(coord[1]), 6)
            })

        return {
            'method': 'pca',
            'points': points
        }
    
    def find_matches(self, query_image_path, gallery_dir, pet_types, top_k=10):
        """
        Find top-k matching images from the gallery using similarity search.
        
        Complete matching pipeline:
        1. Extract query image embedding (custom + MobileNetV2 if enabled)
        2. Scan gallery directories for specified pet types
        3. Extract embeddings for all gallery images in batches
        4. Compute cosine similarity between query and all gallery embeddings
        5. Rank by similarity score (highest first)
        6. Generate thumbnails for top matches
        7. Return results with similarity scores and metadata
        
        Args:
            query_image_path: Path to the query pet image
            gallery_dir: Root directory containing preprocessed images (e.g., c:/xampp/htdocs/mine/Preprocessed/)
            pet_types: Comma-separated pet types to search (e.g., "cat,dog")
            top_k: Number of top matches to return (default: 10)
        
        Returns:
            List of dictionaries with match information:
            [{
                'filename': 'img_xxx.jpg',
                'path': '/full/path/to/image.jpg',
                'similarity': 0.92,
                'type': 'Cat',
                'thumbnail': 'base64_encoded_image_string'
            }, ...]
        
        Note:
            Gallery directories are expected to be named with plural forms:
            - gallery_dir/Cats/ for cat images
            - gallery_dir/Dogs/ for dog images
        """
        self.log(f"Finding matches for: {query_image_path}")
        self.log(f"Gallery directory: {gallery_dir}")
        self.log(f"Pet types: {pet_types}")
        self.log(f"Top K: {top_k}")
        
        # Parse pet types
        pet_types_list = [pt.strip().title() for pt in pet_types.split(',')]
        # If Unknown is declared, restrict search to Unknown folder only
        if 'Unknown' in pet_types_list:
            pet_types_list = ['Unknown']
        self.log(f"Parsed pet types: {pet_types_list}")
        
        # Get query embedding
        try:
            if self.use_mnetv2:
                query_embedding = self.get_combined_embedding(query_image_path)
                self.log("Using combined embedding (custom + MobileNetV2)")
            else:
                query_embedding = self.get_embedding(query_image_path)
                self.log("Using custom model embedding only")
            
            self.log(f"Query embedding shape: {query_embedding.shape}")
            self.log(f"Query embedding stats: mean={query_embedding.mean():.4f}, std={query_embedding.std():.4f}")
        except Exception as e:
            raise Exception(f"Failed to extract query embedding: {str(e)}")
        
        # Collect gallery images
        gallery_images = []
        for pet_type in pet_types_list:
            # Use 'Unknown' (no plural), otherwise pluralize (Cats/Dogs)
            dir_name = 'Unknown' if pet_type == 'Unknown' else pet_type + 's'
            type_dir = os.path.join(gallery_dir, dir_name)
            if not os.path.exists(type_dir):
                self.log(f"Warning: Directory not found: {type_dir}")
                continue
            
            self.log(f"Scanning directory: {type_dir}")
            
            for filename in os.listdir(type_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    img_path = str(Path(os.path.join(type_dir, filename)).resolve())
                    gallery_images.append({
                        'path': img_path,
                        'filename': filename,
                        'type': pet_type
                    })
        
        self.log(f"Found {len(gallery_images)} gallery images")
        
        if len(gallery_images) == 0:
            raise Exception(f"No gallery images found in {gallery_dir}")
        
        gallery_lookup = {item['path']: item for item in gallery_images}
        # Process gallery images in batches for efficiency
        gallery_paths = [item['path'] for item in gallery_images]
        gallery_vectors = []
        
        self.log(f"Processing gallery images in batches of {self.batch_size}...")
        
        # Process in batches
        for i in range(0, len(gallery_paths), self.batch_size):
            batch_paths = gallery_paths[i:i + self.batch_size]
            try:
                batch_embeddings, valid_paths = self.get_embeddings_batch(batch_paths)
                if len(batch_embeddings) == 0:
                    continue
                for emb, valid_path in zip(batch_embeddings, valid_paths):
                    meta = gallery_lookup.get(valid_path)
                    if meta is None:
                        meta = {
                            'path': valid_path,
                            'filename': os.path.basename(valid_path),
                            'type': 'Unknown'
                        }
                    gallery_vectors.append({'embedding': emb, 'meta': meta})
                
                self.log(f"Processed batch {i // self.batch_size + 1}/{(len(gallery_paths) + self.batch_size - 1) // self.batch_size}")
            except Exception as e:
                self.log(f"Error processing batch starting at index {i}: {str(e)}")
                # Fallback to individual processing for this batch
                for path in batch_paths:
                    try:
                        if self.use_mnetv2:
                            emb = self.get_combined_embedding(path)
                        else:
                            emb = self.get_embedding(path)
                        meta = gallery_lookup.get(path)
                        if meta is None:
                            meta = {
                                'path': path,
                                'filename': os.path.basename(path),
                                'type': 'Unknown'
                            }
                        gallery_vectors.append({'embedding': emb, 'meta': meta})
                    except Exception as e2:
                        self.log(f"Error processing {path}: {str(e2)}")
                        continue
        
        if len(gallery_vectors) == 0:
            raise Exception("Failed to extract any gallery embeddings")
        
        self.log(f"Successfully extracted {len(gallery_vectors)} embeddings")
        
        # Compute similarities using vectorized operations
        gallery_embeddings = np.array([entry['embedding'] for entry in gallery_vectors])
        
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        gallery_norms = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute all similarities at once (vectorized)
        similarities = np.dot(gallery_norms, query_norm)
        similarities = (similarities + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
        
        # Create matches list
        matches = []
        for entry, similarity in zip(gallery_vectors, similarities):
            meta = entry['meta']
            matches.append({
                'path': meta['path'],
                'filename': meta['filename'],
                'type': meta['type'],
                'similarity': float(similarity),
                'distance': float(1.0 - similarity)
            })
        
        self.log(f"Successfully computed {len(matches)} similarities")
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top-k matches
        top_k_matches = matches[:top_k]
        
        # Generate thumbnails in parallel for top-k matches
        self.log(f"Generating thumbnails for top {top_k} matches in parallel...")
        top_k_paths = [match['path'] for match in top_k_matches]
        thumbnails = self.generate_thumbnails_parallel(top_k_paths)
        
        self.log(f"Thumbnail generation results: {len([t for t in thumbnails.values() if t])} succeeded, {len([t for t in thumbnails.values() if not t])} failed")
        
        # Build final results with ranks and thumbnails
        top_matches = []
        for rank, match in enumerate(top_k_matches, start=1):
            thumb = thumbnails.get(match['path'])
            if thumb is None:
                self.log(f"WARNING: No thumbnail for {match['path']}")
            
            top_matches.append({
                'rank': rank,
                'path': match['path'],
                'filename': match['filename'],
                'type': match['type'],
                'similarity': match['similarity'],
                'distance': match['distance'],
                'thumb_base64': thumb
            })
        
        self.log(f"Top match similarity: {top_matches[0]['similarity']:.4f}" if top_matches else "No matches")
        embedding_lookup = {entry['meta']['path']: entry['embedding'] for entry in gallery_vectors}
        max_points = self._resolve_embedding_cap(len(top_matches))
        embedding_map = self._build_embedding_map(query_embedding, top_matches, embedding_lookup, max_points=max_points)
        
        return {
            'matches': top_matches,
            'embedding_map': embedding_map
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - Command-line interface for pet image matching.
    
    Usage:
        python compute_matches.py <query_image> <pet_types> <gallery_dir> <top_k> [options]
    
    Required Arguments:
        query_image: Path to the pet image to search for
        pet_types: Comma-separated list of pet types (e.g., "cat,dog")
        gallery_dir: Root directory containing preprocessed pet images
        top_k: Number of top matches to return
    
    Optional Arguments:
        --debug: Enable verbose debug logging
        --no-mnetv2: Disable MobileNetV2 ensemble (use only custom model)
        --mnetv2-weight=X: Set MobileNetV2 ensemble weight (0.0-1.0, default: 0.3)
    
    Output:
        JSON object with structure:
        {
            "ok": true/false,
            "matches": [{
                "rank": 1,
                "path": "/path/to/match.jpg",
                "filename": "img_xxx.jpg",
                "type": "Cat",
                "similarity": 0.92,
                "distance": 0.08,
                "confidence": 92.0,
                "score": 92.0,
                "thumb_base64": "base64_encoded_thumbnail"
            }, ...],
            "debug": ["debug messages"],
            "error": "error message if failed"
        }
    
    Exit Codes:
        0: Success
        1: Invalid arguments or execution error
    """
    
    # Parse command line arguments
    if len(sys.argv) < 5:
        print(json.dumps({
            'ok': False,
            'error': 'Usage: compute_matches.py <query_image> <pet_types> <gallery_dir> <top_k> [--debug]'
        }))
        sys.exit(1)
    
    query_image = sys.argv[1]
    pet_types = sys.argv[2]
    gallery_dir = sys.argv[3]
    top_k = int(sys.argv[4])
    debug = '--debug' in sys.argv
    
    # Get model path (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(script_dir), 'model', 'final_best_model.keras')
    
    result = {
        'ok': False,
        'matches': [],
        'debug': [],
        'attempts': []
    }
    
    try:
        # Initialize engine with optimized settings
        num_workers = cpu_count()
        batch_size = min(32, max(8, num_workers * 2))  # Adaptive batch size
        
        # Parse optional parameters for MobileNetV2 ensemble
        use_mnetv2 = '--no-mnetv2' not in sys.argv
        mnetv2_weight = 0.4 # Default weight for MobileNetV2 embeddings
        
        # Check for custom MobileNetV2 weight
        for arg in sys.argv:
            if arg.startswith('--mnetv2-weight='):
                try:
                    mnetv2_weight = float(arg.split('=')[1])
                    mnetv2_weight = max(0.0, min(1.0, mnetv2_weight))  # Clamp between 0 and 1
                except ValueError:
                    pass
        
        engine = PetMatchingEngine(
            model_path, 
            debug=debug, 
            batch_size=batch_size,
            num_workers=num_workers,
            use_mnetv2=use_mnetv2,
            mnetv2_weight=mnetv2_weight
        )
        
        if debug:
            result['debug'].append(f"Using {num_workers} CPU cores with batch size {batch_size}")
            result['debug'].append(f"MobileNetV2 ensemble: {use_mnetv2}, weight: {mnetv2_weight}")
        
        # Load model
        engine.load_model()
        
        # Load MobileNetV2 as base model if enabled
        if use_mnetv2:
            try:
                engine.load_mnetv2_model()
            except Exception as e:
                engine.use_mnetv2 = False
        
        # Find matches
        payload = engine.find_matches(query_image, gallery_dir, pet_types, top_k)
        matches = payload.get('matches', []) if isinstance(payload, dict) else payload
        embedding_map = payload.get('embedding_map') if isinstance(payload, dict) else None
        
        # Format results for PHP
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'rank': match['rank'],
                'path': match['path'].replace('\\', '/'),  # Convert to forward slashes for PHP
                'filename': match['filename'],
                'type': match['type'],
                'similarity': round(match['similarity'], 4),
                'distance': round(match['distance'], 4),
                'confidence': round(match['similarity'] * 100, 2),  # Convert to percentage
                'score': round(match['similarity'] * 100, 2),  # Same as confidence, for JS compatibility
                'thumb_base64': match['thumb_base64']  # Base64 encoded thumbnail
            })
        
        result['ok'] = True
        result['matches'] = formatted_matches
        result['debug'] = engine.debug_info
        if embedding_map and isinstance(embedding_map, dict):
            sanitized_points = []
            for point in embedding_map.get('points', []):
                cleaned = dict(point)
                path_val = cleaned.get('path')
                if path_val:
                    cleaned['path'] = path_val.replace('\\', '/')
                sanitized_points.append(cleaned)
            result['embedding_map'] = {
                'method': embedding_map.get('method', 'pca'),
                'points': sanitized_points
            }
        else:
            result['embedding_map'] = None
        
    except Exception as e:
        result['error'] = str(e)
        result['debug'] = [f"Exception: {str(e)}"]
        if debug:
            import traceback
            result['debug'].append(traceback.format_exc())
        result['embedding_map'] = None
    
    # Output JSON result
    print(json.dumps(result, indent=2 if debug else None))


if __name__ == '__main__':
    main()
