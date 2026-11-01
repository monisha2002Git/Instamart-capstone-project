"""Part A, task 7 -- pandas cross-check of Q2 and exploratory analysis of completed orders."""
import os
import numpy as np
import pandas as pd

CLEAN = os.path.join("data", "clean")


def main():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))

    comp = orders[orders["status"] == "completed"].copy()
    m = comp.merge(customers[["customer_id", "city"]], on="customer_id") \
            .merge(products[["product_id", "category"]], on="product_id")

    # (a) pivot_table with margins -- grand total must equal the SQL revenue total
    pv = pd.pivot_table(m, values="order_amount", index="city", columns="category",
                        aggfunc="sum", margins=True, margins_name="All")
    grand_total = round(float(pv.loc["All", "All"]), 2)
    print(f"pivot_table grand total: INR {grand_total}  [ref 2998668.80]")

    # groupby named aggregation per city
    agg = m.groupby("city").agg(
        total=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    ).round(2)
    print("\nper-city completed order_amount (sum / mean / count):")
    print(agg)
    print("[ref Bengaluru sum 523643.40 mean 276.47 count 1894; Mumbai sum 479058.50 mean 297.00 count 1613]")

    # (b) distribution statistics
    a = comp["order_amount"]
    mean, median = a.mean(), a.median()
    skew = a.skew()
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_above = int((a > upper_fence).sum())
    n_below = int((a < lower_fence).sum())
    print(f"\norder_amount: mean {mean:.2f}, median {median:.2f}, skewness {skew:.2f}")
    print(f"  Q1 {q1:.2f}, Q3 {q3:.2f}, IQR {iqr:.2f}")
    print(f"  lower fence {lower_fence:.2f}, upper fence {upper_fence:.2f}")
    print(f"  outliers above upper fence: {n_above}; below lower fence: {n_below}")
    print("  [ref mean 269.91, median 135.00, skew 4.61, Q1 64.80, Q3 297.00, upper 645.30, 897 above, 0 below]")

    # correlation matrix
    corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print("\ncorrelation matrix (completed orders):")
    print(corr.round(3))
    # largest absolute off-diagonal correlation
    c = corr.where(~np.eye(len(corr), dtype=bool))
    stacked = c.abs().stack()
    i, j = stacked.idxmax()
    print(f"largest |correlation|: {i} & {j} = {corr.loc[i, j]:.3f}  [ref order_amount/discount_pct ~ -0.095]")
    print("NOTE: a correlation of this size does not establish causation.")


if __name__ == "__main__":
    main()
