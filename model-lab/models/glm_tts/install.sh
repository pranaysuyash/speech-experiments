#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repo"
VENV_DIR="$SCRIPT_DIR/venv"
REPO_URL="https://github.com/zai-org/GLM-TTS.git"
REPO_COMMIT="c5dc7aecc3b4032032d631b271e767893984f821"

echo "=== GLM-TTS Installation ==="
echo "Target commit: $REPO_COMMIT"
echo ""

# Pre-flight: Check usage context
if [ ! -f "$SCRIPT_DIR/requirements.txt" ] || [ ! -d "$SCRIPT_DIR/patches" ]; then
    echo "ERROR: Run this script from the models/glm_tts directory."
    echo "  cd models/glm_tts && ./install.sh"
    exit 1
fi

# Pre-flight: Check system dependencies (minimal set for install, not weights)
echo "[0/5] Checking system dependencies..."

for cmd in git python3 pkg-config; do
    if ! command -v $cmd >/dev/null; then
        echo "ERROR: Missing command '$cmd'"
        exit 1
    fi
done

if ! pkg-config --exists openfst 2>/dev/null; then
    if [ -f /opt/homebrew/include/fst/fst.h ] || [ -f /usr/local/include/fst/fst.h ]; then
        echo "  OpenFST headers: found"
    else
        echo "  OpenFST: not found"
        echo "    Install with: brew install openfst"
        exit 1
    fi
else
    echo "  OpenFST: found (pkg-config)"
fi

# macOS Homebrew setup
if [ -d /opt/homebrew/include ]; then
    export CFLAGS="-I/opt/homebrew/include ${CFLAGS:-}"
    export CXXFLAGS="-I/opt/homebrew/include ${CXXFLAGS:-}"
    export LDFLAGS="-L/opt/homebrew/lib ${LDFLAGS:-}"
    export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

echo ""

# Step 1: Clone and reset repo
echo "[1/5] Setting up repo..."
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$REPO_COMMIT"
else
    rm -rf "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$REPO_COMMIT"
fi

# Strict cleanup prevents stale state
git reset --hard "$REPO_COMMIT"
git clean -ffd

CURRENT_COMMIT=$(git rev-parse HEAD)
if [ "$CURRENT_COMMIT" != "$REPO_COMMIT" ]; then
    echo "ERROR: Commit mismatch! Expected $REPO_COMMIT, got $CURRENT_COMMIT"
    exit 1
fi
echo "Repo ready at $CURRENT_COMMIT"
echo ""

# Step 2: Create virtual environment
echo "[2/5] Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
echo "venv: $VENV_DIR"
echo ""

# Step 3: Install base dependencies
echo "[3/5] Installing Python dependencies..."
$PIP install --upgrade "pip==24.0" "setuptools==75.1.0" "wheel==0.44.0"

# Ensure requirements.txt is strictly compliant
if grep -Eq '^\s*pynini(\s*==|\s*$)' "$SCRIPT_DIR/requirements.txt"; then
  echo "ERROR: requirements.txt must NOT contain pynini. install.sh installs it explicitly."
  exit 1
fi

$PIP install -r "$SCRIPT_DIR/requirements.txt"
echo "Dependencies installed"
echo ""

# Step 4: Apply patches deterministically
echo "[4/5] Applying device-support patches..."
cd "$REPO_DIR"

PATCH_GLOB=("$SCRIPT_DIR/patches"/*.patch)
# Check if glob expanded to existing file
if [ ! -f "${PATCH_GLOB[0]}" ]; then
  echo "ERROR: No patch files found at $SCRIPT_DIR/patches/*.patch"
  exit 1
fi

PATCH_COUNT=0
for patch in "${PATCH_GLOB[@]}"; do
    echo "  Applying: $(basename "$patch")"
    
    # Strict check: fail if patch doesn't apply cleanly
    if ! git apply --check "$patch"; then
       echo "ERROR: Patch does not apply cleanly: $(basename "$patch")"
       exit 1
    fi
    
    git apply "$patch" || {
        echo "ERROR: Patch apply did not run: $(basename "$patch")"
        exit 1
    }
    PATCH_COUNT=$((PATCH_COUNT + 1))
done

echo "$PATCH_COUNT patches applied"
echo ""

# Step 5: Install pynini
echo "[5/5] Installing pynini (requires OpenFST)..."
$PIP install pynini==2.1.7
echo "pynini installed"
echo ""

# Artifacts
echo "=== Installation wrap-up ==="

# Create a provenance receipt placeholder for weights
RECEIPT="$SCRIPT_DIR/ckpt/.source.json"
mkdir -p "$SCRIPT_DIR/ckpt"
if [ ! -f "$RECEIPT" ]; then
  cat > "$RECEIPT" <<'JSON'
{"repo_id":"zai-org/GLM-TTS","note":"Run huggingface-cli download zai-org/GLM-TTS --local-dir ckpt"}
JSON
  echo "Created receipt placeholder: $RECEIPT"
fi

# Freeze environment state
$PIP freeze > "$SCRIPT_DIR/venv.freeze.txt"
echo "Environment frozen to venv.freeze.txt"

echo ""
echo "=== GLM-TTS Installation: end ==="
echo ""
echo "Next step (run from models/glm_tts):"
echo "  huggingface-cli download zai-org/GLM-TTS --local-dir ckpt"
