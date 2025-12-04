#!/usr/bin/env python3
"""Master validation script - runs all validation checks.

This script orchestrates all validation checks and produces a comprehensive report.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_validation_script(script_name, description):
    """Run a validation script and capture results."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)

    script_path = project_root / "scripts" / script_name

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=project_root,
        )

        if result.returncode == 0:
            print(f"\n✓ {description} - PASSED")
            return True
        else:
            print(f"\n⚠️ {description} - FAILED (exit code: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("MASTER VALIDATION SUITE")
    print("=" * 80)
    print(f"Project: SpeckitDLBird")
    print(f"Root: {project_root}")
    print("\nThis will run comprehensive validation checks on:")
    print("  - Data integrity and indexing")
    print("  - Feature extraction and quality")
    print("  - Class balance and distribution")
    print("=" * 80)

    results = {}

    # Run all validation scripts
    validations = [
        ("validate_data.py", "Data Integrity Check"),
        ("validate_features.py", "Feature Quality Check"),
        ("validate_class_balance.py", "Class Balance Analysis"),
    ]

    for script, description in validations:
        passed = run_validation_script(script, description)
        results[description] = passed

    # Summary
    print("\n" + "=" * 80)
    print("MASTER VALIDATION SUMMARY")
    print("=" * 80)

    total_checks = len(results)
    passed_checks = sum(results.values())

    print(f"\nValidation Results: {passed_checks}/{total_checks} checks passed\n")

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status:12} {check}")

    # Final verdict
    print("\n" + "=" * 80)

    if passed_checks == total_checks:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\n📝 Next Steps:")
        print("  1. Review validation reports in artifacts/validation/")
        print("  2. Check VALIDATION_CHECKLIST.md and mark items complete")
        print("  3. Address any warnings or recommendations")
        print("  4. Proceed with audio model improvements")
        return_code = 0
    elif passed_checks >= total_checks * 0.8:
        print("⚠️ MOST VALIDATIONS PASSED (with warnings)")
        print("\n📝 Recommended Actions:")
        print("  1. Review failed checks above")
        print("  2. Address critical issues before proceeding")
        print("  3. Document known issues and workarounds")
        return_code = 0
    else:
        print("❌ MULTIPLE VALIDATIONS FAILED")
        print("\n🚨 Critical Actions Required:")
        print("  1. Fix data integrity issues")
        print("  2. Validate feature extraction pipeline")
        print("  3. Do not proceed with new experiments until fixed")
        return_code = 1

    print(f"\nValidation reports saved to: {project_root / 'artifacts' / 'validation'}")
    print("=" * 80)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
