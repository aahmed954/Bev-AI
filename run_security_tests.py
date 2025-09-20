#!/usr/bin/env python3
"""
Security module test runner
Run comprehensive tests for the advanced security operations center
"""

import subprocess
import sys
import os
from pathlib import Path

def run_security_tests():
    """Run all security module tests"""

    # Get the project root directory
    project_root = Path(__file__).parent
    test_dir = project_root / "tests" / "security"

    print("🔒 Running Advanced Security Operations Center Tests")
    print("=" * 60)

    # Check if test directory exists
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return 1

    # Install test dependencies if needed
    try:
        import pytest
        import pytest_asyncio
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "pytest", "pytest-asyncio", "pytest-cov"
        ], check=True)

    # Run tests with pytest
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--asyncio-mode=auto",  # Auto async mode
        "--cov=src.security",  # Coverage for security modules
        "--cov-report=term-missing",  # Show missing lines
        str(test_dir)
    ]

    print(f"🧪 Running tests in: {test_dir}")
    print(f"📋 Test command: pytest {' '.join(test_args)}")
    print()

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest"
        ] + test_args,
        cwd=project_root,
        check=False
        )

        if result.returncode == 0:
            print("\n✅ All security tests passed!")
            print("🛡️ Advanced Security Operations Center validated")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")

        return result.returncode

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_quick_validation():
    """Run quick validation of security modules"""

    print("🚀 Quick Security Module Validation")
    print("=" * 40)

    # Try importing all security modules
    modules_to_test = [
        "src.security.security_framework",
        "src.security.tactical_intelligence",
        "src.security.defense_automation",
        "src.security.opsec_enforcer",
        "src.security.intel_fusion"
    ]

    success_count = 0

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}: {e}")

    print()
    print(f"📊 Module Import Results: {success_count}/{len(modules_to_test)}")

    if success_count == len(modules_to_test):
        print("🎉 All security modules imported successfully!")
        return 0
    else:
        print("⚠️ Some modules failed to import")
        return 1

def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Security Operations Center Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.quick:
        return run_quick_validation()
    else:
        return run_security_tests()

if __name__ == "__main__":
    sys.exit(main())