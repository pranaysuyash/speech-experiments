#!/bin/bash

# Fix Jupyter Interpreter Issues for Model Lab

echo "🔧 Jupyter Interpreter Fix Script"
echo "================================="
echo ""

# Current directory check
CURRENT_DIR=$(pwd)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$CURRENT_DIR" != "$SCRIPT_DIR" ]; then
    echo "🔄 Changing to script directory..."
    cd "$SCRIPT_DIR"
fi

if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from model-lab directory"
    echo "   Current directory: $CURRENT_DIR"
    echo "   Script directory: $SCRIPT_DIR"
    exit 1
fi

echo "✅ Found model-lab directory"

# Check UV environment
if [ ! -d ".venv" ]; then
    echo "❌ Error: .venv directory not found"
    echo "   Run: uv init"
    exit 1
fi

echo "✅ Found .venv directory"

# Get absolute paths
VENV_PYTHON=$(pwd)/.venv/bin/python
echo "📍 UV Python: $VENV_PYTHON"

# Test that UV Python works
echo "🧪 Testing UV Python..."
$VENV_PYTHON --version

if [ $? -eq 0 ]; then
    echo "✅ UV Python works"
else
    echo "❌ UV Python not working"
    exit 1
fi

# Check if ipykernel is installed
echo "🔍 Checking ipykernel..."
$VENV_PYTHON -c "import ipykernel; print('ipykernel installed')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  ipykernel not found, installing..."
    uv add ipykernel
fi

# Check kernel directory
KERNEL_DIR=".venv/share/jupyter/kernels/python3"
echo "📁 Kernel directory: $KERNEL_DIR"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "🔧 Creating kernel directory..."
    mkdir -p "$KERNEL_DIR"
fi

# Create kernel.json
echo "📝 Creating kernel.json..."
cat > "$KERNEL_DIR/kernel.json" << EOF
{
 "argv": [
  "$(pwd)/.venv/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python 3 (model-lab)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
EOF

echo "✅ kernel.json created"

# Verify kernel.json
echo "🔍 Verifying kernel.json..."
if [ -f "$KERNEL_DIR/kernel.json" ]; then
    echo "✅ kernel.json exists"
    cat "$KERNEL_DIR/kernel.json"
else
    echo "❌ kernel.json not created"
    exit 1
fi

# List all Jupyter kernels
echo ""
echo "📋 Available Jupyter kernels:"
jupyter kernelspec list

# Test that the kernel is valid
echo ""
echo "🧪 Testing kernel validity..."
python -m ipykernel --version

if [ $? -eq 0 ]; then
    echo "✅ ipykernel working"
else
    echo "❌ ipykernel not working"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Restart Jupyter Lab:"
echo "   jupyter lab"
echo ""
echo "2. In Jupyter, select kernel:"
echo "   Kernel -> Change Kernel -> Python 3 (model-lab)"
echo ""
echo "3. Verify correct Python:"
echo "   import sys"
echo "   print(sys.executable)"
echo "   # Should show: $(pwd)/.venv/bin/python"
echo ""
echo "✅ Fix complete!"