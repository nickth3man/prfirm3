#!/usr/bin/env python3
"""
Test runner script for the Virtual PR Firm project.

This script provides a convenient way to run all tests with proper configuration,
coverage reporting, and various test options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for Virtual PR Firm")
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run tests quickly (skip slow tests)"
    )
    parser.add_argument(
        "--lint", 
        action="store_true", 
        help="Run linting checks"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all checks (tests, coverage, linting)"
    )
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific option is given
    if not any([args.unit, args.integration, args.coverage, args.lint, args.all]):
        args.all = True
    
    success = True
    
    # Run linting if requested
    if args.lint or args.all:
        print("\n🔍 Running linting checks...")
        
        # Check if black is available
        try:
            black_result = run_command(
                ["black", "--check", "--diff", "."],
                "Code formatting check (black)"
            )
            if not black_result:
                success = False
        except FileNotFoundError:
            print("⚠️  black not found, skipping formatting check")
        
        # Check if flake8 is available
        try:
            flake8_result = run_command(
                ["flake8", ".", "--max-line-length=88", "--ignore=E203,W503"],
                "Code style check (flake8)"
            )
            if not flake8_result:
                success = False
        except FileNotFoundError:
            print("⚠️  flake8 not found, skipping style check")
    
    # Run tests
    if args.unit or args.integration or args.all:
        print("\n🧪 Running tests...")
        
        # Build pytest command
        pytest_cmd = ["python", "-m", "pytest"]
        
        if args.verbose:
            pytest_cmd.append("-v")
        
        if args.coverage or args.all:
            pytest_cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml"
            ])
        
        if args.fast:
            pytest_cmd.append("-m")
            pytest_cmd.append("not slow")
        
        # Add test directories
        if args.unit:
            pytest_cmd.extend(["tests/test_main.py", "tests/test_flow.py", "tests/test_nodes.py"])
        elif args.integration:
            pytest_cmd.extend(["-m", "integration"])
        else:
            pytest_cmd.append("tests/")
        
        test_result = run_command(pytest_cmd, "Unit and integration tests")
        if not test_result:
            success = False
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All checks passed!")
        print("🎉 Your code is ready for production!")
    else:
        print("❌ Some checks failed!")
        print("🔧 Please fix the issues above before proceeding.")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())