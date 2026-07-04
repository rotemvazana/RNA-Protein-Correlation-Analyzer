# tests/conftest.py
# -----------------
# Pytest configuration file — runs automatically before any test.
#
# The most important job of this file is to patch the `cptac` package
# before any test module is imported. This means tests can run without
# an internet connection or any CPTAC data downloaded, since we replace
# the real cptac with a lightweight mock object.

import sys
from unittest.mock import MagicMock

# Patch cptac before any src/ module is imported.
# This must happen here (in conftest.py) rather than inside individual
# test files, because pytest imports all test modules at startup —
# if data_loader.py tries to `import cptac` during that import phase
# and cptac isn't installed/mocked yet, the whole test run crashes.
sys.modules["cptac"] = MagicMock()
