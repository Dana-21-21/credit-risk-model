import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from src.data_processing import (
    extract_time_features,
    create_aggregate_features
)

import pandas as pd


def test_extract_time_features():

    sample = pd.DataFrame({
        "TransactionStartTime": [
            "2018-11-15 02:18:49"
        ]
    })

    result = extract_time_features(sample)

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns


def test_create_aggregate_features():

    sample = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100, 200, 300]
    })

    result = create_aggregate_features(sample)

    assert "total_transaction_amount" in result.columns
    assert "average_transaction_amount" in result.columns
    assert "transaction_count" in result.columns