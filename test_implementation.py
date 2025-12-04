#!/usr/bin/env python3
"""Quick test script to verify implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.utils.species import normalize_species_name
        from src.data.xeno_canto import index_xeno_canto
        from src.data.cub import index_cub
        from src.features.audio import extract_mfcc_features
        from src.datasets.audio import AudioMFCCDataset, create_species_mapping
        from src.datasets.image import ImageDataset, get_image_transforms
        from src.models.audio_cnn import AudioCNN, count_parameters
        from src.models.audio_vit import AudioViT
        from src.models.image_resnet import ImageResNet
        from src.models.image_vit import ImageViT
        from src.training.trainer import Trainer
        from src.evaluation.metrics import evaluate_model
        from src.evaluation.aggregate import aggregate_results
        from src.utils.splits import create_stratified_splits

        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_species_normalization():
    """Test species name normalization."""
    print("\nTesting species normalization...")

    from src.utils.species import normalize_species_name

    test_cases = [
        ("Parus major (Linnaeus, 1758)", "parus major"),
        ("Red-winged_Blackbird", "red winged blackbird"),
        ("Cardinal", "cardinal"),
    ]

    for input_name, expected in test_cases:
        result = normalize_species_name(input_name)
        if result == expected:
            print(f"  ✓ '{input_name}' -> '{result}'")
        else:
            print(f"  ✗ '{input_name}' -> '{result}' (expected '{expected}')")
            return False

    print("✓ Species normalization tests passed!")
    return True


def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")

    import torch
    from src.models.audio_cnn import AudioCNN, count_parameters
    from src.models.image_resnet import ImageResNet

    try:
        # Audio CNN
        model = AudioCNN(num_classes=10)
        params = count_parameters(model)
        print(f"  ✓ AudioCNN created with {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 3, 40, 130)
        out = model(x)
        assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
        print(f"  ✓ AudioCNN forward pass successful: {x.shape} -> {out.shape}")

        # Image ResNet
        model = ImageResNet(num_classes=10, pretrained=False)
        params = count_parameters(model)
        print(f"  ✓ ImageResNet created with {params:,} parameters")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10), f"Expected shape (2, 10), got {out.shape}"
        print(f"  ✓ ImageResNet forward pass successful: {x.shape} -> {out.shape}")

        print("✓ Model creation tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SpeckitDLBird Implementation Tests")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Species Normalization", test_species_normalization()))
    results.append(("Model Creation", test_model_creation()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
