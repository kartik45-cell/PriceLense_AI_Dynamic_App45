"""Loading, validation and preparation for the food-pricing data."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "id", "week", "center_id", "meal_id", "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured", "num_orders",
}


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    data_dir = Path(data_dir)
    train_path = data_dir / "raw" / "train1.csv"
    meals_path = data_dir / "raw" / "meal_info.csv"
    train = pd.read_csv(train_path)
    missing = REQUIRED_COLUMNS - set(train.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    meals = pd.read_csv(meals_path) if meals_path.exists() else None
    return train, meals


def clean_transactions(train: pd.DataFrame, meals: pd.DataFrame | None = None) -> pd.DataFrame:
    df = train.copy()
    df = df.drop_duplicates(subset="id").dropna(subset=["checkout_price", "base_price", "num_orders"])
    df = df[(df.checkout_price > 0) & (df.base_price > 0) & (df.num_orders >= 0)]
    for col in ("week", "center_id", "meal_id", "emailer_for_promotion", "homepage_featured"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if meals is not None:
        df = df.merge(meals.drop_duplicates("meal_id"), on="meal_id", how="left", validate="m:1")
    df["discount_amount"] = (df["base_price"] - df["checkout_price"]).clip(lower=0)
    df["discount_pct"] = df["discount_amount"] / df["base_price"]
    df["revenue"] = df["checkout_price"] * df["num_orders"]
    return df.sort_values(["week", "center_id", "meal_id"]).reset_index(drop=True)


def save_processed(df: pd.DataFrame, data_dir: str | Path) -> Path:
    path = Path(data_dir) / "processed" / "transactions_clean.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
