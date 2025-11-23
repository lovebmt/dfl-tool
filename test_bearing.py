"""Test bearing dataset loading and distribution"""

import sys

def test_bearing_dataset():
    """Test loading and distributing bearing dataset"""
    print("Testing Bearing Dataset...")
    print("-" * 60)
    
    try:
        from data_utils import BearingDatasetLoader
        
        # Test auto-download and loading
        print("1. Loading bearing dataset (auto-download)...")
        dataset = BearingDatasetLoader()
        
        print(f"   ✓ Train samples: {len(dataset.train_dataset)}")
        print(f"   ✓ Test samples: {len(dataset.test_dataset)}")
        print(f"   ✓ Features: {dataset.num_features}")
        print(f"   ✓ Classes: {dataset.num_classes}")
        
        # Test data sample
        print("\n2. Checking data format...")
        sample_x, sample_y = dataset.train_dataset[0]
        print(f"   ✓ Feature shape: {sample_x.shape}")
        print(f"   ✓ Label: {sample_y.item()}")
        print(f"   ✓ Sample features: {sample_x[:5].tolist()}")
        
        # Test IID distribution
        print("\n3. Testing IID distribution (5 peers)...")
        peer_data = dataset.distribute_data(num_peers=5, distribution_type="iid")
        print(f"   ✓ Created {len(peer_data)} peer datasets")
        for i, (train, test) in enumerate(peer_data):
            print(f"   ✓ Peer {i}: {len(train)} train samples, {len(test)} test samples")
        
        # Test Non-IID distribution
        print("\n4. Testing Non-IID distribution (5 peers)...")
        peer_data = dataset.distribute_data(num_peers=5, distribution_type="non_iid", alpha=0.5)
        print(f"   ✓ Created {len(peer_data)} peer datasets (non-IID)")
        
        # Test label skew distribution
        print("\n5. Testing Label Skew distribution (4 peers)...")
        peer_data = dataset.distribute_data(num_peers=4, distribution_type="label_skew")
        print(f"   ✓ Created {len(peer_data)} peer datasets (label skew)")
        
        print("\n" + "=" * 60)
        print("✓ All bearing dataset tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_bearing():
    """Test model initialization with bearing dataset dimensions"""
    print("\n\nTesting Model with Bearing Dataset...")
    print("-" * 60)
    
    try:
        from model import init_model
        from data_utils import BearingDatasetLoader
        
        # Load dataset to get dimensions
        dataset = BearingDatasetLoader()
        
        print(f"1. Creating model for bearing data...")
        print(f"   Input dim: {dataset.num_features}")
        print(f"   Output dim: {dataset.num_classes}")
        
        model = init_model(
            input_dim=dataset.num_features,
            output_dim=dataset.num_classes,
            device="cpu"
        )
        
        print(f"   ✓ Model created successfully")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        import torch
        sample_x, _ = dataset.train_dataset[0]
        sample_x = sample_x.unsqueeze(0)  # Add batch dimension
        
        output = model(sample_x)
        print(f"   ✓ Input shape: {sample_x.shape}")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output classes: {output.shape[1]}")
        
        print("\n" + "=" * 60)
        print("✓ Model tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dfl_peer_with_bearing():
    """Test DFLPeer with bearing dataset"""
    print("\n\nTesting DFLPeer with Bearing Dataset...")
    print("-" * 60)
    
    try:
        from dfl_peer import DFLPeer
        from data_utils import BearingDatasetLoader
        
        # Load and distribute data
        dataset = BearingDatasetLoader()
        peer_data = dataset.distribute_data(num_peers=3, distribution_type="iid")
        
        train_data, test_data = peer_data[0]
        
        print("1. Creating DFLPeer...")
        peer = DFLPeer(
            peer_id=0,
            train_data=train_data,
            test_data=test_data,
            batch_size=32,
            device="cpu",
            learning_rate=0.001
        )
        
        print(f"   ✓ Peer created with {peer.num_train_samples} train samples")
        
        # Test training
        print("\n2. Testing local training...")
        train_loss = peer.train_local(epochs=1)
        print(f"   ✓ Training completed, loss: {train_loss:.4f}")
        
        # Test evaluation
        print("\n3. Testing evaluation...")
        eval_loss, accuracy = peer.evaluate_local()
        print(f"   ✓ Evaluation completed")
        print(f"   ✓ Eval loss: {eval_loss:.4f}")
        print(f"   ✓ Accuracy: {accuracy:.4f}")
        
        # Test model info
        print("\n4. Testing model export...")
        model_info = peer.get_model_info()
        print(f"   ✓ Model exported")
        print(f"   ✓ Num samples: {model_info['num_samples']}")
        print(f"   ✓ Parameters keys: {len(model_info['parameters'])}")
        
        print("\n" + "=" * 60)
        print("✓ DFLPeer tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Bearing Dataset Integration Tests")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Bearing Dataset Loading", test_bearing_dataset()))
    results.append(("Model Integration", test_model_with_bearing()))
    results.append(("DFLPeer Integration", test_dfl_peer_with_bearing()))
    
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! Bearing dataset integration is ready.")
        print("\nYou can now:")
        print("  1. Start API server: python api.py")
        print("  2. Run bearing examples: python example_bearing.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
