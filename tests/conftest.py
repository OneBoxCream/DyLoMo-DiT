import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def exact_fp32():
    """Some PyTorch builds enable TF32 matrix products on CUDA by default; the tests compare in plain fp32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
