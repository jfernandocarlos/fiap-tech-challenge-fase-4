"""Fixtures compartilhadas."""

from __future__ import annotations

import pandas as pd
import pytest

from credit.data.generate import generate_invalid_batch, generate_reference_dataset


@pytest.fixture
def reference_df() -> pd.DataFrame:
    return generate_reference_dataset(n_samples=500, seed=42)


@pytest.fixture
def invalid_df() -> pd.DataFrame:
    return generate_invalid_batch(seed=99)
