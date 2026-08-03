"""Test configuration.

Forces the mock LLM provider before any app module is imported so the test
suite never hits a live API, regardless of what ``.env`` configures.
"""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "mock"
