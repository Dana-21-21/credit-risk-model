import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans


# =====================================================
# LOAD DATA
# =====================================================

def load_data(filepath):
    """
    Load raw dataset
    """
    return pd.read_csv(filepath)


# =====================================================
# TIME FEATURES
# =====================================================

def extract_time_features(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    df["transaction_hour"] = (
        df["TransactionStartTime"].dt.hour
    )

    df["transaction_day"] = (
        df["TransactionStartTime"].dt.day
    )

    df["transaction_month"] = (
        df["TransactionStartTime"].dt.month
    )

    df["transaction_year"] = (
        df["TransactionStartTime"].dt.year
    )

    return df


# =====================================================
# AGGREGATE FEATURES
# =====================================================

def create_aggregate_features(df):

    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            average_transaction_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            transaction_std=("Amount", "std")
        )
        .reset_index()
    )

    df = df.merge(
        agg_df,
        on="CustomerId",
        how="left"
    )

    return df


# =====================================================
# RFM CALCULATION
# =====================================================

def calculate_rfm(df):

    snapshot_date = (
        pd.to_datetime(
            df["TransactionStartTime"]
        ).max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x:
                (
                    snapshot_date
                    - pd.to_datetime(x).max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    return rfm


# =====================================================
# PROXY TARGET CREATION
# =====================================================

def create_proxy_target(rfm):

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm["cluster"] = (
        kmeans.fit_predict(rfm_scaled)
    )

    cluster_summary = (
        rfm.groupby("cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
    )

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    rfm["is_high_risk"] = (
        rfm["cluster"]
        == high_risk_cluster
    ).astype(int)

    return rfm


# =====================================================
# MERGE TARGET
# =====================================================

def merge_target(df, rfm):

    df = df.merge(
        rfm[
            [
                "CustomerId",
                "is_high_risk"
            ]
        ],
        on="CustomerId",
        how="left"
    )

    return df


# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def build_pipeline():

    numeric_features = [
        "Amount",
        "Value",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "transaction_std"
    ]

    categorical_features = [
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            )
        ]
    )

    return full_pipeline

