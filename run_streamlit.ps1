# ─────────────────────────────────────────────────────────────
#  FA Failure Classifier v6 — One-click launcher
#  Usage:  .\run_streamlit.ps1
# ─────────────────────────────────────────────────────────────

# ── Config ───────────────────────────────────────────────────
# Supports local venv (C:\AIML\efaai_v3) and project venv (.venv_v3)
$LOCAL_VENV = "C:\AIML\efaai_v3"
$PROJ_VENV  = ".venv_v3"
if (Test-Path "$LOCAL_VENV\Scripts\python.exe") {
    $VENV_DIR = $LOCAL_VENV
} else {
    $VENV_DIR = $PROJ_VENV
}
$PIP_EXE    = "$VENV_DIR\Scripts\pip.exe"
$PY_EXE     = "$VENV_DIR\Scripts\python.exe"
$ST_EXE     = "$VENV_DIR\Scripts\streamlit.exe"
$APP_FILE   = "app\streamlit_app.py"
$PORT       = 8501
$REQ_FILE   = "requirements.txt"

# Model files needed for the app
$MODEL_FILES = @(
    "models\v5a\best_model.joblib",
    "models\v5a\tfidf_vectorizer.joblib",
    "models\v5a\label_encoder.joblib",
    "models\v5a\metadata.json"
)

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "   ⚡  FA Failure Classifier v6 (EOS vs Non-EOS)" -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Locate or create venv ────────────────────────────
if (Test-Path $PY_EXE) {
    Write-Host "  ✅  Virtual environment found: $VENV_DIR" -ForegroundColor Green
} else {
    Write-Host "  📦  Virtual environment not found — creating $VENV_DIR ..." -ForegroundColor Yellow

    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "  ❌  Python not found in PATH. Please install Python 3.10+ first." -ForegroundColor Red
        exit 1
    }
    Write-Host "       Using: $($py.Source)" -ForegroundColor Gray

    python -m venv $VENV_DIR
    if (-not (Test-Path $PY_EXE)) {
        Write-Host "  ❌  Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✅  Virtual environment created." -ForegroundColor Green

    Write-Host "  📥  Installing packages from $REQ_FILE ..." -ForegroundColor Yellow
    & $PIP_EXE install --upgrade pip --quiet 2>$null
    & $PIP_EXE install -r $REQ_FILE 2>$null
    Write-Host "  ✅  All packages installed." -ForegroundColor Green
}

# ── Step 2: Quick package check via python import ────────────
Write-Host "  🔍  Checking runtime packages ..." -ForegroundColor Yellow
$checkCmd = "missing=[p for p in ['streamlit','sklearn','pandas','numpy','joblib','scipy','matplotlib','seaborn'] if not __import__('importlib').util.find_spec(p)]; print(','.join(missing) if missing else 'OK')"
$result = & $PY_EXE -c $checkCmd 2>$null

if ($result -and $result -ne "OK") {
    Write-Host "  📥  Missing: $result — installing from $REQ_FILE ..." -ForegroundColor Yellow
    & $PIP_EXE install -r $REQ_FILE 2>$null
    Write-Host "  ✅  Packages installed." -ForegroundColor Green
} else {
    Write-Host "  ✅  All runtime packages present." -ForegroundColor Green
}

# ── Step 3: Check model files ────────────────────────────────
Write-Host "  🔍  Checking model files ..." -ForegroundColor Yellow
$missingModels = @()
foreach ($f in $MODEL_FILES) {
    if (-not (Test-Path $f)) { $missingModels += $f }
}

if ($missingModels.Count -gt 0) {
    Write-Host "  ⚠️   Missing model files:" -ForegroundColor Red
    foreach ($f in $missingModels) {
        Write-Host "       - $f" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "  Run the training notebook first:" -ForegroundColor Yellow
    Write-Host "       notebooks\v5a_NoFullTrain_feat_both_models.ipynb" -ForegroundColor Gray
    Write-Host ""
    $ans = Read-Host "  Continue anyway? (y/N)"
    if ($ans -ne "y" -and $ans -ne "Y") { exit 1 }
} else {
    Write-Host "  ✅  All model files found." -ForegroundColor Green
}

# ── Step 4: Kill any existing Streamlit on this port ─────────
$existing = netstat -ano 2>$null | Select-String ":$PORT\s.*LISTENING"
if ($existing) {
    Write-Host "  🔄  Port $PORT in use — stopping old process ..." -ForegroundColor Yellow
    $existing | ForEach-Object {
        $procId = ($_ -split '\s+')[-1]
        if ($procId -match '^\d+$') {
            Stop-Process -Id ([int]$procId) -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
}

# ── Step 5: Launch ───────────────────────────────────────────
Write-Host ""
Write-Host "  🚀  Launching Streamlit on port $PORT ..." -ForegroundColor Cyan
Write-Host "       http://localhost:$PORT" -ForegroundColor White
Write-Host ""
Write-Host "       Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# Use python -m streamlit — works even when streamlit.exe is absent in the venv
& $PY_EXE -m streamlit run $APP_FILE `
    --server.port=$PORT `
    --server.headless=true `
    --browser.gatherUsageStats=false
