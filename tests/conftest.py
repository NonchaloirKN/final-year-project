"""
Shared pytest configuration.

Every test module builds its own networks and tensors (several ResNet-18 encoders, optimiser state, image batches).
Releasing them after each module keeps a single `pytest` run's memory bounded, so the whole suite can run in one
invocation on a machine with limited free RAM.
"""

import gc

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _release_memory_after_module():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
