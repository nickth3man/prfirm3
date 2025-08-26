#!/usr/bin/env python3
"""
Comprehensive test runner for the project.

This script provides various options for running tests with different configurations,
including unit tests, integration tests, coverage reporting, and performance testing.
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


class TestRunner:
    """Test runner class with various testing configurations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "coverage"
        self.reports_dir = self.project_root / "reports"
        
        # Create necessary directories
        self.coverage_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_command(self, command: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        print(f"Running: {' '.join(command)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            end_time = time.time()
            print(f"Command completed in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error running command: {e}")
            return subprocess.CompletedProcess(command, 1, "", str(e))
    
    def run_unit_tests(self, verbose: bool = False, parallel: bool = False) -> bool:
        """Run unit tests."""
        print("\n" + "="*50)
        print("RUNNING UNIT TESTS")
        print("="*50)
        
        command = ["python", "-m", "pytest", "tests/", "-v" if verbose else ""]
        
        if parallel:
            command.extend(["-n", "auto"])
        
        # Filter for unit tests only
        command.extend(["-m", "unit"])
        
        result = self.run_command([arg for arg in command if arg])
        
        if result.returncode == 0:
            print("✅ Unit tests passed!")
            return True
        else:
            print("❌ Unit tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """Run integration tests."""
        print("\n" + "="*50)
        print("RUNNING INTEGRATION TESTS")
        print("="*50)
        
        command = [
            "python", "-m", "pytest", "tests/",
            "-v" if verbose else "",
            "-m", "integration"
        ]
        
        result = self.run_command([arg for arg in command if arg])
        
        if result.returncode == 0:
            print("✅ Integration tests passed!")
            return True
        else:
            print("❌ Integration tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_api_tests(self, verbose: bool = False) -> bool:
        """Run API tests."""
        print("\n" + "="*50)
        print("RUNNING API TESTS")
        print("="*50)
        
        command = [
            "python", "-m", "pytest", "tests/",
            "-v" if verbose else "",
            "-m", "api"
        ]
        
        result = self.run_command([arg for arg in command if arg])
        
        if result.returncode == 0:
            print("✅ API tests passed!")
            return True
        else:
            print("❌ API tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_coverage_tests(self, verbose: bool = False, html: bool = True) -> bool:
        """Run tests with coverage reporting."""
        print("\n" + "="*50)
        print("RUNNING COVERAGE TESTS")
        print("="*50)
        
        command = [
            "python", "-m", "pytest", "tests/",
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=json:coverage/coverage.json"
        ]
        
        if html:
            command.append("--cov-report=html:coverage/html")
        
        if verbose:
            command.append("-v")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Coverage tests passed!")
            return True
        else:
            print("❌ Coverage tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_performance_tests(self, verbose: bool = False) -> bool:
        """Run performance tests."""
        print("\n" + "="*50)
        print("RUNNING PERFORMANCE TESTS")
        print("="*50)
        
        command = [
            "python", "-m", "pytest", "tests/",
            "--benchmark-only",
            "--benchmark-sort=mean"
        ]
        
        if verbose:
            command.append("-v")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Performance tests passed!")
            return True
        else:
            print("❌ Performance tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_security_tests(self) -> bool:
        """Run security tests."""
        print("\n" + "="*50)
        print("RUNNING SECURITY TESTS")
        print("="*50)
        
        # Run bandit security scanner
        command = ["bandit", "-r", ".", "-f", "json", "-o", "reports/bandit.json"]
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Security tests passed!")
            return True
        else:
            print("❌ Security tests failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_linting(self) -> bool:
        """Run code linting."""
        print("\n" + "="*50)
        print("RUNNING CODE LINTING")
        print("="*50)
        
        # Run flake8
        command = ["flake8", ".", "--output-file=reports/flake8.txt"]
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Code linting passed!")
            return True
        else:
            print("❌ Code linting failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_type_checking(self) -> bool:
        """Run type checking."""
        print("\n" + "="*50)
        print("RUNNING TYPE CHECKING")
        print("="*50)
        
        command = ["mypy", ".", "--html-report", "reports/mypy"]
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Type checking passed!")
            return True
        else:
            print("❌ Type checking failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def run_all_tests(self, verbose: bool = False, parallel: bool = False) -> bool:
        """Run all tests."""
        print("\n" + "="*50)
        print("RUNNING ALL TESTS")
        print("="*50)
        
        start_time = time.time()
        
        # Run all test types
        results = {
            "unit": self.run_unit_tests(verbose, parallel),
            "integration": self.run_integration_tests(verbose),
            "api": self.run_api_tests(verbose),
            "coverage": self.run_coverage_tests(verbose),
            "performance": self.run_performance_tests(verbose),
            "security": self.run_security_tests(),
            "linting": self.run_linting(),
            "type_checking": self.run_type_checking()
        }
        
        end_time = time.time()
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        all_passed = True
        for test_type, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_type.upper():<20} {status}")
            if not passed:
                all_passed = False
        
        print(f"\nTotal time: {end_time - start_time:.2f} seconds")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n💥 SOME TESTS FAILED!")
        
        return all_passed
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> bool:
        """Run a specific test file or test function."""
        print(f"\n" + "="*50)
        print(f"RUNNING SPECIFIC TEST: {test_path}")
        print("="*50)
        
        command = ["python", "-m", "pytest", test_path]
        if verbose:
            command.append("-v")
        
        result = self.run_command(command)
        
        if result.returncode == 0:
            print("✅ Test passed!")
            return True
        else:
            print("❌ Test failed!")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        print("\n" + "="*50)
        print("GENERATING TEST REPORT")
        print("="*50)
        
        report = {
            "timestamp": time.time(),
            "project": "prfirm3",
            "tests": {}
        }
        
        # Run tests and collect results
        test_types = [
            ("unit", self.run_unit_tests),
            ("integration", self.run_integration_tests),
            ("api", self.run_api_tests),
            ("coverage", self.run_coverage_tests),
            ("performance", self.run_performance_tests),
            ("security", self.run_security_tests),
            ("linting", self.run_linting),
            ("type_checking", self.run_type_checking)
        ]
        
        for test_type, test_func in test_types:
            print(f"\nRunning {test_type} tests...")
            start_time = time.time()
            passed = test_func(verbose=False)
            end_time = time.time()
            
            report["tests"][test_type] = {
                "passed": passed,
                "duration": end_time - start_time
            }
        
        # Save report
        report_file = self.reports_dir / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTest report saved to: {report_file}")
        return report
    
    def cleanup(self):
        """Clean up test artifacts."""
        print("\n" + "="*50)
        print("CLEANING UP TEST ARTIFACTS")
        print("="*50)
        
        # Remove test cache
        cache_dirs = [
            ".pytest_cache",
            "__pycache__",
            "tests/__pycache__",
            "tests/.pytest_cache"
        ]
        
        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            if cache_path.exists():
                if cache_path.is_dir():
                    import shutil
                    shutil.rmtree(cache_path)
                else:
                    cache_path.unlink()
                print(f"Removed: {cache_dir}")
        
        print("✅ Cleanup completed!")


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for the project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --verbose         # Run unit tests with verbose output
  python run_tests.py --coverage --html        # Run coverage tests with HTML report
  python run_tests.py --specific tests/test_main.py  # Run specific test file
  python run_tests.py --report                 # Generate comprehensive test report
  python run_tests.py --cleanup                # Clean up test artifacts
        """
    )
    
    # Test type options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--linting", action="store_true", help="Run code linting")
    parser.add_argument("--type-checking", action="store_true", help="Run type checking")
    
    # Test execution options
    parser.add_argument("--specific", type=str, help="Run specific test file or function")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML coverage report")
    
    # Utility options
    parser.add_argument("--report", action="store_true", help="Generate comprehensive test report")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test artifacts")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Handle cleanup
    if args.cleanup:
        runner.cleanup()
        return
    
    # Handle report generation
    if args.report:
        runner.generate_test_report()
        return
    
    # Handle specific test
    if args.specific:
        success = runner.run_specific_test(args.specific, args.verbose)
        sys.exit(0 if success else 1)
    
    # Handle test execution
    success = True
    
    if args.all:
        success = runner.run_all_tests(args.verbose, args.parallel)
    else:
        if args.unit:
            success = runner.run_unit_tests(args.verbose, args.parallel) and success
        if args.integration:
            success = runner.run_integration_tests(args.verbose) and success
        if args.api:
            success = runner.run_api_tests(args.verbose) and success
        if args.coverage:
            success = runner.run_coverage_tests(args.verbose, not args.no_html) and success
        if args.performance:
            success = runner.run_performance_tests(args.verbose) and success
        if args.security:
            success = runner.run_security_tests() and success
        if args.linting:
            success = runner.run_linting() and success
        if args.type_checking:
            success = runner.run_type_checking() and success
        
        # If no specific test type was selected, run unit tests by default
        if not any([args.unit, args.integration, args.api, args.coverage, 
                   args.performance, args.security, args.linting, args.type_checking]):
            success = runner.run_unit_tests(args.verbose, args.parallel)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()