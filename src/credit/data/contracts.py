"""Contratos de qualidade de dados com Pandera.

Regras rígidas:
1. age entre 18 e 75, sem nulos
2. income > 0, sem nulos
3. customer_id único (sem duplicatas)
Extras: loan_amount mínimo e debt_ratio em [0, 1].
"""

from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

from credit.config import load_params


def build_credit_schema() -> DataFrameSchema:
    """Monta o schema Pandera a partir dos limiares do ``params.yaml``."""
    params = load_params()["validation"]
    return DataFrameSchema(
        columns={
            "customer_id": Column(str, nullable=False, unique=True),
            "age": Column(
                int,
                checks=[
                    Check.ge(params["age_min"], error="age_below_minimum"),
                    Check.le(params["age_max"], error="age_above_maximum"),
                ],
                nullable=False,
            ),
            "income": Column(
                float,
                checks=[Check.gt(params["income_min"], error="income_too_low")],
                nullable=False,
                coerce=True,
            ),
            "employment_years": Column(float, nullable=False, coerce=True),
            "loan_amount": Column(
                float,
                checks=[Check.ge(params["loan_amount_min"], error="loan_amount_too_low")],
                nullable=False,
                coerce=True,
            ),
            "loan_duration_months": Column(int, nullable=False),
            "debt_ratio": Column(
                float,
                checks=[
                    Check.ge(0.0, error="debt_ratio_below_zero"),
                    Check.le(1.0, error="debt_ratio_above_one"),
                ],
                nullable=False,
                coerce=True,
            ),
            "num_credit_lines": Column(int, checks=[Check.ge(1)], nullable=False),
            "late_payments": Column(int, checks=[Check.ge(0)], nullable=False),
            "home_ownership": Column(int, checks=[Check.isin([0, 1, 2])], nullable=False),
            "purpose": Column(int, checks=[Check.isin([0, 1, 2, 3])], nullable=False),
            "default": Column(int, checks=[Check.isin([0, 1])], nullable=True),
        },
        coerce=True,
        strict=False,
    )


class DataContractError(Exception):
    """Levantada quando o lote viola o contrato de dados."""

    def __init__(self, message: str, failure_cases: list[str] | None = None):
        super().__init__(message)
        self.failure_cases = failure_cases or []


def validate_dataframe(df, *, raise_on_error: bool = True):
    """Valida um DataFrame contra o contrato.

    Returns:
        DataFrame validado se OK.

    Raises:
        DataContractError: se ``raise_on_error`` e houver violação.
    """
    schema = build_credit_schema()
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = []
        if hasattr(exc, "failure_cases") and exc.failure_cases is not None:
            cases = (
                exc.failure_cases.astype(str)
                .apply(lambda row: " | ".join(row), axis=1)
                .head(20)
                .tolist()
            )
        message = f"Contrato de dados violado ({len(cases)} casos amostrados)."
        if raise_on_error:
            raise DataContractError(message, failure_cases=cases) from exc
        return None
