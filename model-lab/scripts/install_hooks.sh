#!/usr/bin/env bash
set -euo pipefail

echo "Installing git hooks..."

HOOK_PATH=".git/hooks/pre-push"
SCRIPT_PATH="scripts/check_backend.sh"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found!"
    exit 1
fi

mkdir -p .git/hooks

cat > "$HOOK_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
$SCRIPT_PATH
EOF

chmod +x "$HOOK_PATH"

echo "✅ Installed pre-push hook calling $SCRIPT_PATH"
