"""Simple test script to verify DFL tool setup"""

import sys
import importlib


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    modules = [
        'torch',
        'torchvision',
        'numpy',
        'fastapi',
        'uvicorn',
        'pydantic',
        'sklearn',
    ]
    
    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - NOT FOUND")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️  Missing modules: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required modules found")
        return True


def test_local_modules():
    """Test that local modules can be imported"""
    print("\nTesting local modules...")
    
    modules = [
        'config',
        'model',
        'data_utils',
        'dfl_peer',
        'topology',
        'messages',
        'peer_worker',
        'coordinator',
        'api'
    ]
    
    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - ERROR: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All local modules imported successfully")
        return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from model import init_model, get_model_size_bytes
        from topology import RingTopology
        from messages import create_model_message, create_control_message
        
        # Test model
        print("  - Creating model...")
        model = init_model()
        size = get_model_size_bytes(model)
        print(f"    Model size: {size} bytes")
        
        # Test topology
        print("  - Creating topology...")
        topology = RingTopology(5, {1})
        neighbors = topology.get_neighbors(0)
        print(f"    Peer 0 neighbors: {neighbors}")
        
        # Test messages
        print("  - Creating messages...")
        msg = create_control_message("START_ROUND", round_id=1)
        print(f"    Control message: {msg.cmd}")
        
        print("\n✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"\n✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DFL Tool - Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    print()
    
    results.append(("Local Modules", test_local_modules()))
    print()
    
    results.append(("Basic Functionality", test_basic_functionality()))
    print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! DFL tool is ready to use.")
        print("\nTo start the API server, run:")
        print("  python api.py")
        print("\nTo run examples, run:")
        print("  python examples.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
