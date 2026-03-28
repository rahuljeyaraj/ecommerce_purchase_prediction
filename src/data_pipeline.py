import pandas as pd


FILE_PATH = "../data/2019-Nov.csv"
OUTPUT_PATH = "../data/model_ready.csv"
USE_ROW_LIMIT = False
ROW_LIMIT = 100000


def load_data(file_path: str, use_row_limit: bool = True, row_limit: int = 100000) -> pd.DataFrame:
    print("Loading data...")

    if use_row_limit:
        df = pd.read_csv(file_path, nrows=row_limit)
        print(f"Loaded first {row_limit} rows")
    else:
        df = pd.read_csv(file_path)
        print("Loaded full dataset")

    print("Raw shape:", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")

    needed_cols = [
        "event_time",
        "event_type",
        "product_id",
        "price",
        "user_id",
        "user_session",
    ]

    df = df[needed_cols].copy()
    print("Shape after selecting needed columns:", df.shape)

    df = df.dropna(subset=["user_session"])
    print("Shape after dropping missing user_session:", df.shape)

    df["event_time"] = pd.to_datetime(df["event_time"])
    print("Converted event_time to datetime")

    return df


def create_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating event counts per session...")

    session_event_counts = (
        df.groupby(["user_session", "event_type"])
        .size()
        .unstack(fill_value=0)
    )

    print("Event counts shape:", session_event_counts.shape)
    return session_event_counts


def create_session_duration(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating session duration feature...")

    session_time = df.groupby("user_session")["event_time"].agg(["min", "max"])
    session_time["session_duration_sec"] = (
        session_time["max"] - session_time["min"]
    ).dt.total_seconds()

    session_time = session_time[["session_duration_sec"]]

    print("Session duration shape:", session_time.shape)
    return session_time


def create_unique_products(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating unique products feature...")

    session_unique_products = (
        df.groupby("user_session")["product_id"]
        .nunique()
        .to_frame("unique_products")
    )

    print("Unique products shape:", session_unique_products.shape)
    return session_unique_products


def create_avg_price(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating average price feature...")

    session_avg_price = (
        df.groupby("user_session")["price"]
        .mean()
        .to_frame("avg_price")
    )

    print("Average price shape:", session_avg_price.shape)
    return session_avg_price


def combine_features(
    session_event_counts: pd.DataFrame,
    session_duration: pd.DataFrame,
    unique_products: pd.DataFrame,
    avg_price: pd.DataFrame,
) -> pd.DataFrame:
    print("Combining all features...")

    session_df = session_event_counts.join(session_duration, how="left")
    session_df = session_df.join(unique_products, how="left")
    session_df = session_df.join(avg_price, how="left")

    print("Combined session_df shape:", session_df.shape)
    return session_df


def create_target(session_df: pd.DataFrame) -> pd.DataFrame:
    print("Creating target column...")

    if "purchase" not in session_df.columns:
        session_df["purchase"] = 0

    session_df["target"] = (session_df["purchase"] > 0).astype(int)

    session_df = session_df.fillna(0)

    print("Target value counts:")
    print(session_df["target"].value_counts())

    return session_df


def create_model_ready_data(session_df: pd.DataFrame) -> pd.DataFrame:
    print("Creating model-ready dataframe...")

    model_df = session_df.reset_index().drop(columns=["user_session"], errors="ignore")

    print("Model-ready shape:", model_df.shape)
    return model_df


def save_data(df: pd.DataFrame, output_path: str) -> None:
    print(f"Saving processed data to {output_path} ...")
    df.to_csv(output_path, index=False)
    print("Saved successfully.")


def main() -> None:
    df = load_data(FILE_PATH, USE_ROW_LIMIT, ROW_LIMIT)
    df = clean_data(df)

    session_event_counts = create_event_counts(df)
    session_duration = create_session_duration(df)
    unique_products = create_unique_products(df)
    avg_price = create_avg_price(df)

    session_df = combine_features(
        session_event_counts,
        session_duration,
        unique_products,
        avg_price,
    )

    session_df = create_target(session_df)
    model_df = create_model_ready_data(session_df)
    save_data(model_df, OUTPUT_PATH)

    print("\nPipeline completed.")
    print(model_df.head())


if __name__ == "__main__":
    main()