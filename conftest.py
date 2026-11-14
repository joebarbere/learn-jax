"""Pytest glue: every module's test_exercises.py gets an `impl` fixture.

`impl` is that module's exercises.py, or solutions.py when LEARNJAX_SOLUTIONS=1 (how `make check`
verifies the reference answers). Run tests with --import-mode=importlib (set in pyproject.toml) because
every module has a file called test_exercises.py.
"""
import pytest

from learnjax.course import load_exercises


@pytest.fixture
def impl(request):
    return load_exercises(request.path.parent)
