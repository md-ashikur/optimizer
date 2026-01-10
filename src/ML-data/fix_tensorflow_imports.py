"""
Fix for TensorFlow/Keras Import Issues in VS Code
Ensures proper imports work in all environments
"""

# SOLUTION 1: Import tensorflow first, then keras
# This is the recommended approach for TensorFlow 2.x

try:
    # Primary import method (TensorFlow 2.x)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    print("✓ TensorFlow/Keras imported successfully (tensorflow.keras)")
except ImportError as e:
    try:
        # Fallback for older installations
        import keras
        from keras import layers, models, optimizers, callbacks
        print("✓ Keras imported successfully (standalone keras)")
    except ImportError:
        print("✗ ERROR: Neither tensorflow.keras nor standalone keras found")
        print("  Install with: pip install tensorflow")
        raise

# SOLUTION 2: Verify installation
def verify_tensorflow():
    """Verify TensorFlow installation"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")
        print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

# SOLUTION 3: VS Code specific fixes
"""
If VS Code still shows import errors:

1. Check Python interpreter:
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose the .venv environment

2. Reload VS Code:
   - Press Ctrl+Shift+P
   - Type "Developer: Reload Window"

3. Install Python extension:
   - Install "Python" by Microsoft from extensions

4. Install type stubs (optional):
   pip install tensorflow-stubs

5. Update VS Code settings (settings.json):
   {
     "python.analysis.extraPaths": [
       ".venv/Lib/site-packages"
     ],
     "python.languageServer": "Pylance"
   }
"""

# SOLUTION 4: Update all neural network scripts
CORRECT_IMPORT_PATTERN = """
# Correct import pattern for all scripts
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
"""

INCORRECT_IMPORT_PATTERN = """
# AVOID these patterns:
import tensorflow.keras  # May cause issues
from keras import ...     # Only if standalone keras installed
"""

if __name__ == "__main__":
    print("="*80)
    print("TENSORFLOW/KERAS IMPORT FIX")
    print("="*80)
    
    print("\nVerifying installation...")
    if verify_tensorflow():
        print("\n✓ All imports working correctly!")
        print("\nIf VS Code still shows warnings, try:")
        print("  1. Reload window (Ctrl+Shift+P → 'Developer: Reload Window')")
        print("  2. Select correct Python interpreter (.venv)")
        print("  3. Install tensorflow-stubs: pip install tensorflow-stubs")
    else:
        print("\n✗ Installation issues detected")
        print("\nTry reinstalling:")
        print("  pip uninstall tensorflow keras")
        print("  pip install tensorflow")
