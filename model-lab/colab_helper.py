#!/usr/bin/env python3
"""
Colab Upload Helper
Helps upload and run the Colab compatibility test notebook.
"""

import os
import webbrowser
from pathlib import Path

def open_colab_notebook():
    """Open the Colab compatibility test notebook in browser."""
    notebook_path = Path(__file__).parent / "model_lab_colab_test.ipynb"

    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return

    print("🌐 Opening Colab notebook...")
    print("📋 Instructions:")
    print("1. Upload the notebook to Colab")
    print("2. Change runtime to GPU: Runtime → Change runtime type → GPU")
    print("3. Run all cells: Runtime → Run all")
    print("4. Check results at the bottom")

    # Try to open in default browser
    colab_url = "https://colab.research.google.com/"
    try:
        webbrowser.open(colab_url)
        print(f"✅ Opened {colab_url} in browser")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"Please manually visit: {colab_url}")

    print(f"\n📁 Notebook location: {notebook_path.absolute()}")

def main():
    print("🚀 Model-Lab Colab Test Helper")
    print("=" * 40)

    open_colab_notebook()

    print("\n📋 Manual Steps:")
    print("1. Go to https://colab.research.google.com/")
    print("2. Click 'Upload' tab")
    print("3. Upload: model_lab_colab_test.ipynb")
    print("4. Change runtime: Runtime → Change runtime type → GPU")
    print("5. Run all cells: Runtime → Run all")

if __name__ == "__main__":
    main()