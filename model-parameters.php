<?php
set_time_limit(300);

function tokenize_command($command) {
    $command = trim((string)$command);
    if ($command === '') { return []; }
    $tokens = [];
    if (preg_match_all('/("([^"]*)"|\'([^\']*)\'|\S+)/', $command, $matches)) {
        foreach ($matches[0] as $token) {
            if ($token === '') { continue; }
            $first = $token[0];
            $last = substr($token, -1);
            if (($first === '"' && $last === '"') || ($first === "'" && $last === "'")) {
                $tokens[] = substr($token, 1, -1);
            } else {
                $tokens[] = $token;
            }
        }
    }
    return $tokens ?: [$command];
}

function format_command_parts($parts) {
    $escaped = [];
    foreach ($parts as $token) {
        if ($token === null || $token === '') { continue; }
        if (preg_match('/[\s"\']/', $token)) {
            $escaped[] = '"' . str_replace('"', '\\"', $token) . '"';
        } else {
            $escaped[] = $token;
        }
    }
    return implode(' ', $escaped);
}

function detect_python311_command() {
    static $cached = false;
    if ($cached !== false) { return $cached; }

    $candidates = [];
    $envCmd = getenv('PYTHON311_CMD');
    if ($envCmd) { $candidates[] = tokenize_command($envCmd); }
    $candidates[] = ['py', '-3.11'];
    $candidates[] = ['python3.11'];
    $candidates[] = ['python311'];

    $osFamily = defined('PHP_OS_FAMILY') ? PHP_OS_FAMILY : PHP_OS;
    if (stripos($osFamily, 'Windows') !== false) {
        $localApp = getenv('LOCALAPPDATA');
        if ($localApp) {
            $candidates[] = [$localApp . DIRECTORY_SEPARATOR . 'Programs' . DIRECTORY_SEPARATOR . 'Python' . DIRECTORY_SEPARATOR . 'Python311' . DIRECTORY_SEPARATOR . 'python.exe'];
        }
    }
    $candidates[] = ['python'];

    foreach ($candidates as $parts) {
        if (!$parts) { continue; }
        $cmd = format_command_parts($parts);
        if (!$cmd) { continue; }
        $output = @shell_exec($cmd . ' --version 2>&1');
        if ($output && preg_match('/Python\s+3\.11\./', $output)) {
            $cached = $parts;
            return $parts;
        }
    }

    $cached = null;
    return null;
}

$cacheReport = __DIR__ . DIRECTORY_SEPARATOR . 'cache' . DIRECTORY_SEPARATOR . 'Parameters' . DIRECTORY_SEPARATOR . 'report.json';
$report = null;
$runOutput = null;
$runError = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['generate_report'])) {
    $pythonParts = detect_python311_command();
    $pythonCmd = $pythonParts ? format_command_parts($pythonParts) : 'python';

    $script = __DIR__ . DIRECTORY_SEPARATOR . 'python' . DIRECTORY_SEPARATOR . 'explain_parameters.py';
    if (!file_exists($script)) {
        $runError = 'Missing python/explain_parameters.py';
    } else {
        $cmd = $pythonCmd . ' ' . escapeshellarg($script) . ' --max-pairs 200 --examples-per-type 1';
        $runOutput = shell_exec($cmd . ' 2>&1');
        if ($runOutput) {
            $firstBrace = strpos($runOutput, '{');
            $candidate = $firstBrace !== false ? substr($runOutput, $firstBrace) : $runOutput;
            $json = json_decode($candidate, true);
            if ($json && isset($json['ok']) && $json['ok'] && isset($json['report'])) {
                $report = $json['report'];
            } else {
                $runError = $json['error'] ?? ('Failed to parse report JSON. Raw output: ' . substr($runOutput, 0, 600));
            }
        } else {
            $runError = 'No output from python script.';
        }
    }
}

if ($report === null && file_exists($cacheReport)) {
    $raw = file_get_contents($cacheReport);
    if ($raw !== false) {
        $parsed = json_decode($raw, true);
        if (is_array($parsed)) {
            $report = $parsed;
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Parameters - Fur-Get Me Not</title>
    <link rel="stylesheet" href="css/styles.css">
    <link rel="stylesheet" href="css/components.css">
    <style>
        .params-wrap { max-width: 1100px; margin: 0 auto; padding: 0 16px 42px; }
        .params-card { background:#fff; border:1px solid #d9e2f2; border-radius:22px; padding:18px; box-shadow:0 12px 40px rgba(34,69,120,0.10); margin-top:16px; }
        .params-title { font-size:1.2rem; font-weight:800; color:#1b3d7f; margin:0 0 10px; }
        .params-muted { color:#4a5a7b; font-size:0.92rem; line-height:1.45; }
        .params-btn { background:#3867d6; color:#fff; border:none; border-radius:14px; padding:10px 14px; font-weight:700; cursor:pointer; }
        .params-btn:hover { filter:brightness(1.05); }
        .params-badge { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; border:1px solid #dbe4fb; background:#f4f7ff; color:#223a7b; font-weight:700; font-size:0.82rem; }
        .params-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; }
        .params-table { width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border-radius:14px; border:1px solid #d9e2f2; }
        .params-table th, .params-table td { padding:10px 10px; text-align:left; font-size:0.9rem; }
        .params-table th { background:#f4f7ff; color:#223a7b; font-weight:800; }
        .params-table td { border-top:1px solid #edf2ff; color:#334155; }
        .params-kpi { display:flex; gap:10px; flex-wrap:wrap; margin-top:8px; }
        .kpi { background:#f9fbff; border:1px solid #dbe4fb; border-radius:16px; padding:10px 12px; min-width:170px; }
        .kpi .k { color:#4a5a7b; font-size:0.78rem; font-weight:700; }
        .kpi .v { color:#1b3d7f; font-size:1.05rem; font-weight:900; margin-top:2px; }
        .hm-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin-top:10px; margin-bottom:20px; }
        .hm-img { width:100%; border-radius:16px; border:1px solid #d9e2f2; background:#f4f7ff; }
        .hm-caption { font-size:0.82rem; color:#4a5a7b; margin-top:6px; line-height:1.35; }
        .err { color:#9b1c1c; background:#fff5f5; border:1px solid #fecaca; padding:10px 12px; border-radius:14px; }
    </style>
</head>
<body>
    <div class="bg-paws"></div>
    <header>
        <div class="header-bar"></div>
        <a href="index.php" class="back-arrow">
            <img src="assets/How-it-Works/back-arrow.png" alt="Back">
        </a>
        <div class="container">
            <div class="subpage-header">
                <img src="assets/Logos/interface-setting-app-widget--Streamline-Core.png" alt="Icon" class="subpage-icon">
                <h1 class="subpage-title">Model Parameters</h1>
                <div class="subpage-subpill">Baseline vs Proposed (Normal vs Lowlight)</div>
            </div>
        </div>
    </header>

    <main class="params-wrap">
        <div class="params-card">
            <h2 class="params-title">Generate / Refresh Report</h2>
            <p class="params-muted">
                This page summarizes what both models are "looking at" (parameters/features) when deciding if two images are similar or dissimilar.
                It uses gradient-based heatmaps + a lowlight scenario, then maps attention into interpretable regions (ears/eyes/muzzle/fur/body/background).
            </p>
            <form method="post" style="margin-top:10px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <button class="params-btn" type="submit" name="generate_report" value="1">Generate report</button>
                <span class="params-badge">Output: cache/Parameters/report.json</span>
                <?php if ($report && isset($report['generated_at'])): ?>
                    <span class="params-badge">Generated: <?php echo htmlspecialchars($report['generated_at']); ?></span>
                <?php endif; ?>
            </form>
            <?php if ($runError): ?>
                <div class="err" style="margin-top:10px;"><?php echo htmlspecialchars($runError); ?></div>
            <?php endif; ?>
        </div>

        <?php if (!$report): ?>
            <div class="params-card">
                <h2 class="params-title">No report yet</h2>
                <p class="params-muted">
                    Click <strong>Generate report</strong>. If it fails, ensure your Python env has packages from requirements.txt and that PHP can run shell_exec.
                </p>
            </div>
        <?php else: ?>
            <div class="params-card">
                <h2 class="params-title">Model Files & Trainable Parameters</h2>
                <div class="params-grid">
                    <div>
                        <div class="params-badge">Baseline</div>
                        <table class="params-table" style="margin-top:10px;">
                            <tr><th>Model</th><td><?php echo htmlspecialchars($report['models']['baseline']['type'] ?? ''); ?></td></tr>
                            <tr><th>Path</th><td><?php echo htmlspecialchars($report['models']['baseline']['path'] ?? ''); ?></td></tr>
                            <tr><th>Param count</th><td><?php echo htmlspecialchars((string)($report['models']['baseline']['param_count'] ?? '')); ?></td></tr>
                            <tr><th>Input</th><td><?php echo htmlspecialchars(json_encode($report['models']['baseline']['input'] ?? [])); ?></td></tr>
                        </table>
                    </div>
                    <div>
                        <div class="params-badge">Proposed</div>
                        <table class="params-table" style="margin-top:10px;">
                            <tr><th>Model</th><td><?php echo htmlspecialchars($report['models']['proposed']['type'] ?? ''); ?></td></tr>
                            <tr><th>Path</th><td><?php echo htmlspecialchars($report['models']['proposed']['path'] ?? ''); ?></td></tr>
                            <tr><th>Param count</th><td><?php echo htmlspecialchars((string)($report['models']['proposed']['param_count'] ?? '')); ?></td></tr>
                            <tr><th>Input</th><td><?php echo htmlspecialchars(json_encode($report['models']['proposed']['input'] ?? [])); ?></td></tr>
                            <tr><th>Ensemble</th><td><?php echo htmlspecialchars('use_mnetv2=' . (($report['models']['proposed']['use_mnetv2'] ?? false) ? 'true' : 'false') . ', weight=' . ($report['models']['proposed']['mnetv2_weight'] ?? '')); ?></td></tr>
                        </table>
                    </div>
                </div>
                <p class="params-muted" style="margin-top:10px;">
                    Note: the "parameters" below refer to interpretable feature regions (what the model attends to), not raw weight tensors.
                </p>
            </div>

            <div class="params-card">
                <h2 class="params-title">Interpretable Parameters (Feature Proxies)</h2>
                <table class="params-table">
                    <tr>
                        <th>Parameter</th>
                        <th>Definition (what we measure)</th>
                    </tr>
                    <?php foreach (($report['parameters_definition'] ?? []) as $p): ?>
                        <tr>
                            <td><?php echo htmlspecialchars($p['name'] ?? ''); ?></td>
                            <td><?php echo htmlspecialchars($p['definition'] ?? ''); ?></td>
                        </tr>
                    <?php endforeach; ?>
                </table>
                <p class="params-muted" style="margin-top:10px;">
                    Why proxy parameters: the Siamese training does not supervise ear/eye/nose labels, so we infer which visual areas drive similarity using heatmaps.
                </p>
            </div>

            <?php
            $scenarios = $report['scenarios'] ?? [];
            foreach (['normal' => 'Normal', 'lowlight' => 'Lowlight'] as $key => $label):
                $sc = $scenarios[$key] ?? null;
                if (!$sc) continue;
                $thr = $sc['thresholds'] ?? [];
                $attr = $sc['region_attribution_avg_percent'] ?? [];
            ?>
                <div class="params-card">
                    <h2 class="params-title"><?php echo htmlspecialchars($label); ?> Scenario — Performance & Parameter Reliance</h2>

                    <div class="params-grid">
                        <div>
                            <div class="params-badge">Baseline</div>
                            <?php $b = $thr['baseline'] ?? []; $bm = $b['metrics_at_best'] ?? []; $bc = $b['confusion'] ?? []; ?>
                            <div class="params-kpi">
                                <div class="kpi"><div class="k">Best threshold</div><div class="v"><?php echo htmlspecialchars((string)round($b['best_threshold'] ?? 0, 4)); ?></div></div>
                                <div class="kpi"><div class="k">F1</div><div class="v"><?php echo htmlspecialchars((string)round(($bm['f1'] ?? 0)*100, 2)); ?>%</div></div>
                                <div class="kpi"><div class="k">Accuracy</div><div class="v"><?php echo htmlspecialchars((string)round(($bm['accuracy'] ?? 0)*100, 2)); ?>%</div></div>
                            </div>
                            <table class="params-table" style="margin-top:10px;">
                                <tr><th>TP</th><td><?php echo htmlspecialchars((string)($bc['TP'] ?? 0)); ?></td></tr>
                                <tr><th>TN</th><td><?php echo htmlspecialchars((string)($bc['TN'] ?? 0)); ?></td></tr>
                                <tr><th>FP</th><td><?php echo htmlspecialchars((string)($bc['FP'] ?? 0)); ?></td></tr>
                                <tr><th>FN</th><td><?php echo htmlspecialchars((string)($bc['FN'] ?? 0)); ?></td></tr>
                            </table>

                            <div class="params-muted" style="margin-top:10px; font-weight:800; color:#223a7b;">Avg saliency by parameter (%)</div>
                            <table class="params-table" style="margin-top:8px;">
                                <?php foreach (($attr['baseline'] ?? []) as $k2 => $v2): ?>
                                    <tr><th><?php echo htmlspecialchars($k2); ?></th><td><?php echo htmlspecialchars((string)round($v2, 2)); ?>%</td></tr>
                                <?php endforeach; ?>
                            </table>
                        </div>

                        <div>
                            <div class="params-badge">Proposed</div>
                            <?php $p = $thr['proposed'] ?? []; $pm = $p['metrics_at_best'] ?? []; $pc = $p['confusion'] ?? []; ?>
                            <div class="params-kpi">
                                <div class="kpi"><div class="k">Best threshold</div><div class="v"><?php echo htmlspecialchars((string)round($p['best_threshold'] ?? 0, 4)); ?></div></div>
                                <div class="kpi"><div class="k">F1</div><div class="v"><?php echo htmlspecialchars((string)round(($pm['f1'] ?? 0)*100, 2)); ?>%</div></div>
                                <div class="kpi"><div class="k">Accuracy</div><div class="v"><?php echo htmlspecialchars((string)round(($pm['accuracy'] ?? 0)*100, 2)); ?>%</div></div>
                            </div>
                            <table class="params-table" style="margin-top:10px;">
                                <tr><th>TP</th><td><?php echo htmlspecialchars((string)($pc['TP'] ?? 0)); ?></td></tr>
                                <tr><th>TN</th><td><?php echo htmlspecialchars((string)($pc['TN'] ?? 0)); ?></td></tr>
                                <tr><th>FP</th><td><?php echo htmlspecialchars((string)($pc['FP'] ?? 0)); ?></td></tr>
                                <tr><th>FN</th><td><?php echo htmlspecialchars((string)($pc['FN'] ?? 0)); ?></td></tr>
                            </table>

                            <div class="params-muted" style="margin-top:10px; font-weight:800; color:#223a7b;">Avg saliency by parameter (%)</div>
                            <table class="params-table" style="margin-top:8px;">
                                <?php foreach (($attr['proposed'] ?? []) as $k2 => $v2): ?>
                                    <tr><th><?php echo htmlspecialchars($k2); ?></th><td><?php echo htmlspecialchars((string)round($v2, 2)); ?>%</td></tr>
                                <?php endforeach; ?>
                            </table>
                        </div>
                    </div>

                    <div style="margin-top:14px;" class="params-muted">
                        <strong>Interpretation:</strong> when lowlight is enabled, drops in ears/eyes/muzzle percentages usually indicate the model struggles to see facial cues
                        and may rely more on fur/body texture or background (context leakage), which increases FP/FN.
                    </div>

                    <?php
                    $ex = $sc['examples'] ?? [];
                    foreach (['baseline' => 'Baseline', 'proposed' => 'Proposed'] as $mk => $mlabel):
                        $items = $ex[$mk] ?? [];
                        if (!$items) continue;
                    ?>
                        <div style="margin-top:48px; margin-bottom:12px;" class="params-muted"><strong><?php echo htmlspecialchars($mlabel); ?> heatmap examples (TP/TN/FP/FN)</strong></div>
                        <?php foreach ($items as $it): ?>
                            <div style="background:#f9fbff; border:2px solid #d9e2f2; border-radius:20px; padding:20px; margin-bottom:24px;">
                                <div style="display:flex; gap:12px; align-items:center; margin-bottom:16px; flex-wrap:wrap;">
                                    <div class="params-badge" style="font-size:0.88rem;"><?php echo htmlspecialchars(($it['kind'] ?? '') . ' — score=' . round($it['score'] ?? 0, 4)); ?></div>
                                    <div class="hm-caption" style="margin:0; flex-grow:1;">
                                        <strong>Label:</strong> <?php echo htmlspecialchars(($it['label_same'] ?? false) ? 'Same' : 'Different'); ?> | 
                                        <strong>Pred:</strong> <?php echo htmlspecialchars(($it['pred_same'] ?? false) ? 'Same' : 'Different'); ?> | 
                                        <strong>Threshold:</strong> <?php echo htmlspecialchars((string)round($it['threshold'] ?? 0, 4)); ?>
                                    </div>
                                </div>
                                <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
                                    <div>
                                        <img class="hm-img" src="<?php echo htmlspecialchars($it['heatmap1'] ?? ''); ?>" alt="Heatmap 1" style="width:100%; border-radius:14px;">
                                        <div class="hm-caption" style="margin-top:10px; font-weight:600;">Image 1</div>
                                        <div class="hm-caption"><?php echo htmlspecialchars($it['img1'] ?? ''); ?></div>
                                    </div>
                                    <div>
                                        <img class="hm-img" src="<?php echo htmlspecialchars($it['heatmap2'] ?? ''); ?>" alt="Heatmap 2" style="width:100%; border-radius:14px;">
                                        <div class="hm-caption" style="margin-top:10px; font-weight:600;">Image 2</div>
                                        <div class="hm-caption"><?php echo htmlspecialchars($it['img2'] ?? ''); ?></div>
                                    </div>
                                </div>
                            </div>
                        <?php endforeach; ?>
                    <?php endforeach; ?>
                </div>
            <?php endforeach; ?>

            <div class="params-card">
                <h2 class="params-title">Factors Affecting Baseline vs Proposed (Before Conclusion)</h2>
                <table class="params-table">
                    <tr><th>Factor</th><th>Why it matters</th></tr>
                    <?php foreach (($report['performance_factors'] ?? []) as $f): ?>
                        <tr>
                            <td><?php echo htmlspecialchars($f); ?></td>
                            <td><?php echo htmlspecialchars('This factor changes the visibility of the visual cues used as parameters, and can shift the optimal threshold or increase confusion (FP/FN).'); ?></td>
                        </tr>
                    <?php endforeach; ?>
                </table>
            </div>

            <div class="params-card">
                <h2 class="params-title">Major Gaps in the Context of Those Parameters</h2>
                <table class="params-table">
                    <tr><th>Gap</th></tr>
                    <?php foreach (($report['gaps_in_parameters_context'] ?? []) as $g): ?>
                        <tr><td><?php echo htmlspecialchars($g); ?></td></tr>
                    <?php endforeach; ?>
                </table>
            </div>

            <div class="params-card">
                <h2 class="params-title">Architectural Challenges (Move out of Chapter 5)</h2>
                <table class="params-table">
                    <tr><th>Challenge</th></tr>
                    <?php foreach (($report['architectural_challenges'] ?? []) as $g): ?>
                        <tr><td><?php echo htmlspecialchars($g); ?></td></tr>
                    <?php endforeach; ?>
                </table>
            </div>

            <div class="params-card">
                <h2 class="params-title">Conclusion Mapping (Parameters → Results)</h2>
                <p class="params-muted">
                    Use the table above (Avg saliency by parameter) + the FP/FN heatmap examples to justify:
                    <br>• what parameters each model depends on (ears/eyes/muzzle vs fur/body vs background)
                    <br>• how lowlight shifts those parameters (visibility loss → more context leakage)
                    <br>• why Proposed vs Baseline differs (CapsNet+attention and ensemble changes which cues dominate)
                </p>
            </div>
        <?php endif; ?>
    </main>

    <script src="js/page-transitions.js"></script>
</body>
</html>
