"""Shared fixtures. Tests run from any folder: ``python main.py test`` or ``pytest``."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_arm_rl.config import Config  # noqa: E402
from dual_arm_rl.env.scene import Scene  # noqa: E402

BASIC, MODERATE, FULL = 0, 1, 2


@pytest.fixture(scope="session")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def scene(cfg):
    return Scene(cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def stage(cfg, k):
    return cfg.curriculum.stages[k]
