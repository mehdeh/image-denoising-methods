"""
Test script to verify the modular structure works correctly.

This script performs basic sanity checks on all modules without
requiring heavy computations or dataset downloads.
"""

import torch
import sys


def test_imports():
    """Test that all modules can be imported"""
    print("="*80)
    print("Testing Module Imports")
    print("="*80)
    
    try:
        from denoisers.ideal_denoiser import ideal_denoiser
        print("✓ denoisers.ideal_denoiser imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ideal_denoiser: {e}")
        return False
    
    try:
        from denoisers.edm_denoiser import (
            load_edm_model, edm_denoise, compute_score_gradient,
            gradient_ascent_denoise, load_pretrained_edm
        )
        print("✓ denoisers.edm_denoiser imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import edm_denoiser: {e}")
        return False
    
    try:
        from utils.noise_utils import add_gaussian_noise
        print("✓ utils.noise_utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import noise_utils: {e}")
        return False
    
    try:
        from utils.image_utils import load_cifar10_dataset, normalize_for_display
        print("✓ utils.image_utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import image_utils: {e}")
        return False
    
    try:
        from utils.visualization import create_labeled_figure
        print("✓ utils.visualization imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import visualization: {e}")
        return False
    
    print()
    return True


def test_noise_utils():
    """Test noise utility functions"""
    print("="*80)
    print("Testing Noise Utils")
    print("="*80)
    
    from utils.noise_utils import add_gaussian_noise
    
    # Test basic functionality
    images = torch.randn(2, 3, 32, 32)
    
    # Test with different sigma values
    for sigma in [0, 0.5, 1.0, 2.0]:
        noisy = add_gaussian_noise(images, sigma)
        assert noisy.shape == images.shape, f"Shape mismatch for sigma={sigma}"
        
        if sigma == 0:
            assert torch.allclose(noisy, images), "Zero noise should return identical images"
        else:
            assert not torch.allclose(noisy, images), f"Noisy images should differ for sigma={sigma}"
    
    print("✓ add_gaussian_noise works correctly")
    print()
    return True


def test_image_utils():
    """Test image utility functions"""
    print("="*80)
    print("Testing Image Utils")
    print("="*80)
    
    from utils.image_utils import normalize_for_display
    
    # Test normalization
    images = torch.randn(5, 3, 32, 32)
    normalized = normalize_for_display(images)
    
    assert normalized.shape == images.shape, "Shape mismatch after normalization"
    assert normalized.min() >= 0.0, "Normalized images should be >= 0"
    assert normalized.max() <= 1.0, "Normalized images should be <= 1"
    
    print("✓ normalize_for_display works correctly")
    print()
    return True


def test_ideal_denoiser():
    """Test ideal denoiser with synthetic data"""
    print("="*80)
    print("Testing Ideal Denoiser")
    print("="*80)
    
    from denoisers.ideal_denoiser import ideal_denoiser
    from utils.noise_utils import add_gaussian_noise
    
    # Create small synthetic dataset
    train_images = torch.randn(100, 3, 32, 32)
    test_image = torch.randn(1, 3, 32, 32)
    
    # Add noise
    sigma = 2.0
    noisy = add_gaussian_noise(test_image, sigma)
    
    # Denoise
    with torch.no_grad():
        denoised = ideal_denoiser(noisy, sigma, train_images)
    
    assert denoised.shape == test_image.shape, "Output shape mismatch"
    print(f"✓ Ideal denoiser works (input shape: {noisy.shape}, output shape: {denoised.shape})")
    
    # Test batch processing
    batch_noisy = add_gaussian_noise(train_images[:5], sigma)
    with torch.no_grad():
        batch_denoised = ideal_denoiser(batch_noisy, sigma, train_images)
    
    assert batch_denoised.shape == batch_noisy.shape, "Batch output shape mismatch"
    print(f"✓ Ideal denoiser batch processing works (batch size: {batch_noisy.shape[0]})")
    print()
    return True


def test_package_init():
    """Test that package __init__.py exports work correctly"""
    print("="*80)
    print("Testing Package Exports")
    print("="*80)
    
    # Test denoisers package
    from denoisers import ideal_denoiser, edm_denoise
    print("✓ denoisers package exports work")
    
    # Test utils package
    from utils import add_gaussian_noise, load_cifar10_dataset, normalize_for_display
    print("✓ utils package exports work")
    
    print()
    return True


def test_generate_figure1_imports():
    """Test that generate_edm_figure1.py can import from modules"""
    print("="*80)
    print("Testing generate_edm_figure1.py Integration")
    print("="*80)
    
    try:
        # This will test if the imports in generate_edm_figure1.py work
        import generate_edm_figure1
        print("✓ generate_edm_figure1.py imports successfully")
        
        # Check that it has the main function
        assert hasattr(generate_edm_figure1, 'main'), "main() function not found"
        assert hasattr(generate_edm_figure1, 'generate_figure1'), "generate_figure1() function not found"
        print("✓ Required functions exist in generate_edm_figure1.py")
        
    except Exception as e:
        print(f"✗ Failed to import generate_edm_figure1.py: {e}")
        return False
    
    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MODULAR STRUCTURE TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Noise Utils", test_noise_utils),
        ("Image Utils", test_image_utils),
        ("Ideal Denoiser", test_ideal_denoiser),
        ("Package Exports", test_package_init),
        ("generate_edm_figure1.py", test_generate_figure1_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✓ All tests passed! The modular structure is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

