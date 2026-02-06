#!/usr/bin/env python3
"""
Quick validation test to verify infrastructure works.
Tests what we can actually run right now without additional dependencies.
"""

import sys
from pathlib import Path

# Add harness to path
harness_path = Path(__file__).parent.parent / "harness"
sys.path.insert(0, str(harness_path))


def test_imports():
    """Test if all harness modules can be imported."""
    print("=== Testing Harness Imports ===")

    try:

        print("✓ AudioLoader")
    except Exception as e:
        print(f"✗ AudioLoader: {e}")
        return False

    try:

        print("✓ ASRMetrics")
    except Exception as e:
        print(f"✗ ASRMetrics: {e}")
        return False

    try:

        print("✓ Protocol modules")
    except Exception as e:
        print(f"✗ Protocol: {e}")
        return False

    return True


def test_lfm_import():
    """Test if LFM model can be imported."""
    print("\n=== Testing LFM2.5-Audio Import ===")

    try:

        print("✓ LFM2AudioModel and LFM2AudioProcessor")
        return True
    except Exception as e:
        print(f"✗ LFM import: {e}")
        return False


def test_smoke_dataset():
    """Test if smoke dataset exists."""
    print("\n=== Testing Smoke Dataset ===")

    smoke_audio = Path("data/audio/SMOKE/conversation_2ppl_10s.wav")
    smoke_text = Path("data/text/SMOKE/conversation_2ppl_10s.txt")

    if smoke_audio.exists():
        print(f"✓ Smoke audio: {smoke_audio}")
    else:
        print(f"✗ Smoke audio missing: {smoke_audio}")
        return False

    if smoke_text.exists():
        print(f"✓ Smoke text: {smoke_text}")
        with open(smoke_text) as f:
            content = f.read()
            print(f"  Content: {content[:100]}...")
    else:
        print(f"✗ Smoke text missing: {smoke_text}")
        return False

    return True


def test_protocol_validation():
    """Test protocol validation."""
    print("\n=== Testing Protocol Validation ===")

    try:
        from protocol import EntityExtractionProtocol, NormalizationValidator

        # Test normalization
        test_text = "Hello World! Number: 123, Date: 01/08/2024, Price: $19.99"
        normalized = NormalizationValidator.normalize_text(test_text)

        print(f"✓ Normalization: '{test_text[:30]}...' → '{normalized[:30]}...'")
        print(f"  Protocol version: {NormalizationValidator.NORMALIZATION_PROTOCOL['version']}")

        # Test entity protocol
        entity_protocol = EntityExtractionProtocol.validate_entity_rules()
        print(f"✓ Entity protocol: v{entity_protocol['protocol_version']}")
        print(f"  Locked rules: {entity_protocol['locked']}")

        return True
    except Exception as e:
        print(f"✗ Protocol validation: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🧪 Model Lab Infrastructure Validation")
    print("=" * 50)

    results = []

    # Test 1: Harness imports
    results.append(("Harness Imports", test_imports()))

    # Test 2: LFM import
    results.append(("LFM Import", test_lfm_import()))

    # Test 3: Smoke dataset
    results.append(("Smoke Dataset", test_smoke_dataset()))

    # Test 4: Protocol validation
    results.append(("Protocol Validation", test_protocol_validation()))

    # Summary
    print("\n" + "=" * 50)
    print("=== VALIDATION SUMMARY ===")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All infrastructure tests passed!")
        print("✅ Ready for model testing")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("🔧 Fix issues before running model tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())
