"""Smoke test for the package itself: imports cleanly and exposes a version."""

import nanotse


def test_package_imports() -> None:
    assert hasattr(nanotse, "__version__")


def test_version_is_string() -> None:
    assert isinstance(nanotse.__version__, str)
    assert nanotse.__version__.count(".") >= 1
