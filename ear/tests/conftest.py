import pytest

from core.context import Context


@pytest.fixture
def ctx(tmp_path):
    return Context(workdir=tmp_path)
