"""Part A, task 2 -- audit the raw exports.
Loads the three raw tables and prints shape, dtypes, per-column missingness and the
specific audit figures the brief asks for. Read-only: writes nothing."""
import os
import pandas as pd

RAW = os.path.join("data", "raw")


def pct(n, d):
    return round(100.0 * n / d, 2) if d else 0.0


def missing_report(df, name):
    print(f"\n=== {name}: shape={df.shape} ===")
    print("dtypes:")
    for c in df.columns:
        print(f"  {c}: {df[c].dtype}")
    print("missing (count / pct):")
    for c in df.columns:
        # empty string counts as missing for this raw audit
        blank = df[c].isna() | (df[c].astype(str).str.strip() == "")
        n = int(blank.sum())
        print(f"  {c}: {n} / {pct(n, len(df))}%")


def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False


def main():
    customers = pd.read_csv(os.path.join(RAW, "customers.csv"), dtype=str, keep_default_na=False)
    products = pd.read_csv(os.path.join(RAW, "products.csv"), dtype=str, keep_default_na=False)
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype=str, keep_default_na=False)

    for df, name in [(customers, "customers.csv"), (products, "products.csv"), (orders, "orders.csv")]:
        missing_report(df, name)

    print("\n=== AUDIT FIGURES ===")

    # orders
    n_rows = len(orders)
    n_exact_dup = int(orders.duplicated(keep="first").sum())
    n_pending = int((orders["status"] == "pending").sum())
    n_distinct_oid = orders["order_id"].nunique()
    n_bad_amount = int((~orders["order_amount"].map(is_number)).sum())
    deliv = orders["delivery_minutes"]
    n_neg_deliv = int(deliv.map(lambda x: is_number(x) and float(x) < 0).sum())
    n_blank_deliv = int((deliv.str.strip() == "").sum())

    print(f"orders.csv rows: {n_rows}")
    print(f"exact duplicate rows: {n_exact_dup}")
    print(f"rows with status pending: {n_pending}")
    print(f"distinct order_id: {n_distinct_oid}")
    print(f"non-numeric order_amount: {n_bad_amount}")
    print(f"negative delivery_minutes: {n_neg_deliv}")
    print(f"blank delivery_minutes: {n_blank_deliv}")

    # customers
    age = customers["age"]
    n_blank_age = int((age.str.strip() == "").sum())
    n_age_out = int(age.map(lambda x: is_number(x) and not (10 <= float(x) <= 100)).sum())
    n_city_spellings = customers["city"].nunique()
    gender_nonblank = customers.loc[customers["gender"].str.strip() != "", "gender"]
    n_gender_spellings = gender_nonblank.nunique()
    n_blank_gender = int((customers["gender"].str.strip() == "").sum())

    print(f"blank age: {n_blank_age}")
    print(f"age outside 10-100 (non-blank): {n_age_out}")
    print(f"distinct raw city spellings: {n_city_spellings}")
    print(f"distinct non-blank raw gender spellings: {n_gender_spellings}")
    print(f"blank gender: {n_blank_gender}")


if __name__ == "__main__":
    main()
