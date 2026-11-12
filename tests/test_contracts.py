"""Testes do contrato Pandera."""

from __future__ import annotations

import pandas as pd
import pytest

from credit.data.contracts import DataContractError, validate_dataframe


def test_reference_passes_contract(reference_df: pd.DataFrame) -> None:
    validated = validate_dataframe(reference_df)
    assert len(validated) == len(reference_df)


def test_invalid_batch_is_blocked(invalid_df: pd.DataFrame) -> None:
    with pytest.raises(DataContractError) as exc_info:
        validate_dataframe(invalid_df)
    assert exc_info.value.failure_cases


def test_null_income_fails(reference_df: pd.DataFrame) -> None:
    bad = reference_df.copy()
    bad.loc[0, "income"] = None
    with pytest.raises(DataContractError):
        validate_dataframe(bad)


def test_underage_fails(reference_df: pd.DataFrame) -> None:
    bad = reference_df.copy()
    bad.loc[0, "age"] = 16
    with pytest.raises(DataContractError):
        validate_dataframe(bad)
