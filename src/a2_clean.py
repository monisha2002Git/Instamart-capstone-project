"""Part A, tasks 3-5 -- clean customers and orders, then store.
Writes data/clean/{customers,products,orders}.csv and data/instamart.db
(tables customers, products, orders; order_ts stored as 'YYYY-MM-DD HH:MM:SS' text).
Every rule prints the number of rows/values it changed, for the README."""
import os
import sqlite3
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")
CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]


def clean_customers(products):
    df = pd.read_csv(os.path.join(RAW, "customers.csv"), dtype=str, keep_default_na=False)

    # (a) city: strip + title case
    raw_city_spellings = df["city"].nunique()
    df["city"] = df["city"].str.strip().str.title()
    print(f"[customers] city standardised: {raw_city_spellings} raw spellings -> {df['city'].nunique()} cities")

    # (b) gender -> male / female / unknown
    def map_gender(g):
        g = (g or "").strip().lower()
        if g in ("male", "m"):
            return "male"
        if g in ("female", "f"):
            return "female"
        return "unknown"
    n_unknown_before = int((df["gender"].str.strip() == "").sum())
    df["gender"] = df["gender"].map(map_gender)
    print(f"[customers] gender mapped to male/female/unknown; blank->unknown affected {n_unknown_before} blanks")

    # (c) age: out of 10-100 -> missing, then fill with city median
    age = pd.to_numeric(df["age"], errors="coerce")
    n_blank = int(age.isna().sum())
    out_of_range = age.notna() & ~age.between(10, 100)
    n_out = int(out_of_range.sum())
    age = age.mask(out_of_range)
    df["age"] = age
    n_missing_before_fill = int(df["age"].isna().sum())
    df["age"] = df.groupby("city")["age"].transform(lambda s: s.fillna(s.median()))
    print(f"[customers] age: {n_blank} blank + {n_out} out-of-range -> {n_missing_before_fill} missing filled "
          f"with city median; missing now {int(df['age'].isna().sum())}")

    # (d) signup_date -> datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d")

    return df


def clean_orders(products):
    df = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype=str, keep_default_na=False)
    n0 = len(df)

    # 1. order_ts -> datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"], format="%Y-%m-%d %H:%M:%S")

    # 2. drop exact duplicate rows (compare on the original string columns before further edits)
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"[orders] removed {before - len(df)} exact duplicate rows ({before} -> {len(df)})")

    # 3. keep only the latest order_ts record per order_id
    before = len(df)
    df = df.sort_values("order_ts").groupby("order_id", as_index=False).tail(1)
    df = df.sort_values("order_id").reset_index(drop=True)
    print(f"[orders] collapsed duplicate order_id records (kept latest order_ts): {before} -> {len(df)} rows")
    print(f"[orders]   statuses after collapse: {df['status'].value_counts().to_dict()}")

    # numeric helpers
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")

    # 4. order_amount -> numeric; unparseable -> NaN; fill from product price
    amount = pd.to_numeric(df["order_amount"], errors="coerce")
    n_bad = int(amount.isna().sum())
    price_map = dict(zip(products["product_id"], products["price_inr"].astype(float)))
    price = df["product_id"].map(price_map)
    filled = (price * df["quantity"] * (1 - df["discount_pct"] / 100)).round(2)
    df["order_amount"] = amount.fillna(filled)
    print(f"[orders] order_amount: {n_bad} unparseable values recomputed from price*qty*(1-disc); "
          f"missing now {int(df['order_amount'].isna().sum())}")

    # 5. delivery_minutes: negative -> missing; completed -> fill with category median; cancelled -> leave missing
    deliv = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    n_neg = int((deliv < 0).sum())
    deliv = deliv.mask(deliv < 0)
    df["delivery_minutes"] = deliv
    cat_map = dict(zip(products["product_id"], products["category"]))
    df["_category"] = df["product_id"].map(cat_map)

    completed = df["status"] == "completed"
    n_missing_completed_before = int(df.loc[completed, "delivery_minutes"].isna().sum())
    cat_median = (df[completed].groupby("_category")["delivery_minutes"]
                  .transform("median"))
    fill_vals = df["delivery_minutes"].copy()
    fill_vals.loc[completed] = df.loc[completed, "delivery_minutes"].fillna(cat_median)
    df["delivery_minutes"] = fill_vals
    df = df.drop(columns="_category")
    print(f"[orders] delivery_minutes: {n_neg} negatives set missing; "
          f"{n_missing_completed_before} missing among completed filled with category median; "
          f"missing among completed now {int(df.loc[completed, 'delivery_minutes'].isna().sum())}; "
          f"missing among cancelled {int(df.loc[df['status']=='cancelled', 'delivery_minutes'].isna().sum())}")

    completed_sum = round(float(df.loc[completed, "order_amount"].sum()), 2)
    print(f"[orders] final rows {len(df)} (raw {n0}); completed order_amount sum = INR {completed_sum}")
    return df


def main():
    os.makedirs(CLEAN, exist_ok=True)
    products = pd.read_csv(os.path.join(RAW, "products.csv"), dtype=str, keep_default_na=False)
    products["price_inr"] = products["price_inr"].astype(int)

    customers = clean_customers(products)
    orders = clean_orders(products)

    # ---- write clean CSVs ----
    cust_out = customers.copy()
    cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")
    cust_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)

    products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)

    ord_out = orders.copy()
    ord_out["order_ts"] = ord_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ord_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

    # ---- write SQLite ----
    if os.path.exists(DB):
        os.remove(DB)
    conn = sqlite3.connect(DB)
    cust_out.to_sql("customers", conn, index=False)
    products.to_sql("products", conn, index=False)
    ord_out.to_sql("orders", conn, index=False)
    conn.close()

    print("\n[customers] city counts:", customers["city"].value_counts().to_dict())
    print("[customers] gender counts:", customers["gender"].value_counts().to_dict())
    print(f"[customers] age range: {customers['age'].min()} - {customers['age'].max()}")
    print("\nWrote data/clean/*.csv and data/instamart.db")


if __name__ == "__main__":
    main()
