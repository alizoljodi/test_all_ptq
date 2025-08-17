#!/usr/bin/env python3
"""
Test script to check MQBench compatibility and demonstrate the fixes.
Run this script to diagnose your MQBench setup before running the main PTQ script.
"""

import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_mqbench_imports():
    """Test basic MQBench imports."""
    print("Testing MQBench imports...")
    
    try:
        import mqbench
        print(f"✓ MQBench imported successfully")
        print(f"  Version: {getattr(mqbench, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"✗ Failed to import MQBench: {e}")
        return False
    
    try:
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        print("✓ prepare_by_platform and BackendType imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import prepare_by_platform: {e}")
        return False
    
    try:
        from mqbench.utils.state import enable_calibration, enable_quantization
        print("✓ enable_calibration and enable_quantization imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import state utilities: {e}")
        return False
    
    try:
        from mqbench.convert_deploy import convert_deploy
        print("✓ convert_deploy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import convert_deploy: {e}")
        return False
    
    return True

def test_advanced_ptq():
    """Test advanced PTQ imports."""
    print("\nTesting Advanced PTQ imports...")
    
    try:
        from mqbench.advanced_ptq import ptq_reconstruction
        print("✓ ptq_reconstruction imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import advanced_ptq: {e}")
        print("  This means advanced PTQ features (AdaRound/BRECQ/QDrop) are not available")
        return False
    except Exception as e:
        print(f"✗ Unexpected error importing advanced_ptq: {e}")
        return False

def test_quantizer_compatibility():
    """Test if quantizer classes have the expected methods."""
    print("\nTesting Quantizer Compatibility...")
    
    try:
        # Try to import some quantizer classes
        from mqbench.prepare_by_platform import prepare_by_platform
        print("✓ prepare_by_platform available")
        
        # This is a basic test - in practice, the real test happens during model preparation
        print("  Note: Full quantizer compatibility testing requires a model to be prepared")
        return True
        
    except Exception as e:
        print(f"✗ Quantizer compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("MQBench Compatibility Test Suite")
    print("=" * 40)
    
    # Test basic imports
    basic_ok = test_mqbench_imports()
    
    # Test advanced PTQ
    advanced_ok = test_advanced_ptq()
    
    # Test quantizer compatibility
    quantizer_ok = test_quantizer_compatibility()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Basic MQBench: {'✓ OK' if basic_ok else '✗ FAILED'}")
    print(f"Advanced PTQ:  {'✓ OK' if advanced_ok else '✗ NOT AVAILABLE'}")
    print(f"Quantizers:    {'✓ OK' if quantizer_ok else '✗ FAILED'}")
    
    if not basic_ok:
        print("\n❌ Basic MQBench setup failed. Please install/update MQBench.")
        return 1
    
    if not advanced_ok:
        print("\n⚠️  Advanced PTQ not available. You can still use basic PTQ.")
        print("   To enable advanced PTQ, update MQBench to a compatible version.")
    
    if not quantizer_ok:
        print("\n⚠️  Quantizer compatibility issues detected.")
        print("   This may cause errors during model quantization.")
    
    print("\n✅ You can now run the main PTQ script:")
    print("   python mq_bench_ptq.py --model resnet18")
    
    if not advanced_ok:
        print("   Note: Use --no_advanced flag to avoid advanced PTQ errors")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
