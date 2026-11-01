"""Part A, task 5 -- validate clean data against the brief's reference checkpoints.
Asserts well over eight checkpoints and prints exactly ALL CHECKPOINTS PASSED when all hold.
Reads the clean CSVs and the SQLite database."""
import os
import sqlite3
import pandas as pd

CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")


def main():
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    # ---- customers (task 3) ----
    city_counts = customers["city"].value_counts().to_dict()
    check("cities == 6", customers["city"].nunique() == 6)
    check("Pune 227", city_counts.get("Pune") == 227)
    check("Delhi 210", city_counts.get("Delhi") == 210)
    check("Bengaluru 197", city_counts.get("Bengaluru") == 197)
    gender_counts = customers["gender"].value_counts().to_dict()
    check("female 610", gender_counts.get("female") == 610)
    check("male 560", gender_counts.get("male") == 560)
    check("unknown 30", gender_counts.get("unknown") == 30)
    check("0 missing ages", customers["age"].isna().sum() == 0)
    check("ages within 18-60", customers["age"].between(18, 60).all())

    # ---- orders (task 4) ----
    check("12309 orders", len(orders) == 12309)
    check("unique order_id", orders["order_id"].nunique() == 12309)
    check("11110 completed", (orders["status"] == "completed").sum() == 11110)
    check("1199 cancelled", (orders["status"] == "cancelled").sum() == 1199)
    check("no pending", (orders["status"] == "pending").sum() == 0)
    check("0 missing order_amount", orders["order_amount"].isna().sum() == 0)
    completed = orders["status"] == "completed"
    check("0 missing delivery among completed", orders.loc[completed, "delivery_minutes"].isna().sum() == 0)
    check("1199 missing delivery among cancelled",
          orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum() == 1199)
    check("completed amount sum 2998668.80",
          round(float(orders.loc[completed, "order_amount"].sum()), 2) == 2998668.80)

    # ---- database has the three tables ----
    conn = sqlite3.connect(DB)
    tbls = set(pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn)["name"])
    conn.close()
    check("db has customers/products/orders", {"customers", "products", "orders"} <= tbls)

    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{passed}/{len(checks)} checkpoints passed")
    assert passed == len(checks), "Some checkpoints failed"
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
