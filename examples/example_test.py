import pytest

def run_tests():
    result = pytest.main(["--tb=short", "../tests/"])
    if result == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


run_tests()