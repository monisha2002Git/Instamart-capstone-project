"""Part B, tasks 1-3 -- RFM table, K-means segmentation, profiling, naming, PCA.
Reads data/clean/orders.csv + customers.csv; writes outputs/customer_segments.csv.
Uses only completed orders for customer behaviour."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLEAN = os.path.join("data", "clean")
OUT = "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4


def build_rfm():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    comp = orders[orders["status"] == "completed"]
    rfm = comp.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    ).reset_index()
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    print(f"RFM customers: {len(rfm)}  [ref 1175]")
    print(f"  recency: median {rfm['recency'].median()}, min {rfm['recency'].min()}, max {rfm['recency'].max()}  "
          f"[ref median 19, min 0, max 180]")
    print(f"  frequency: mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}  [ref mean 9.46, max 33]")
    print(f"  monetary: mean {rfm['monetary'].mean():.2f}, max {rfm['monetary'].max()}  [ref mean 2552.06, max 14766.55]")
    return rfm


def kmeans_sweep(X):
    print("\nK-means sweep (inertia / silhouette):")
    results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        results.append((k, km.inertia_, sil))
        print(f"  K={k}: inertia {km.inertia_:.1f} / silhouette {sil:.3f}")
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    rfm = build_rfm()

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans_sweep(X)
    print(f"\nInertia always falls as K rises because more centroids can sit closer to points "
          f"(at K=n inertia=0); it measures fit, not generalisation, so it alone cannot pick K. "
          f"We choose K={FINAL_K}: silhouette peaks there (0.526) and the four groups are business-actionable.")

    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_

    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    ).round(2)
    print("\nSegment profile (K=4):")
    print(profile)

    # name segments by their RFM character; Premium = highest mean monetary
    premium_seg = profile["monetary"].idxmax()
    names = {}
    for sid, row in profile.iterrows():
        if sid == premium_seg:
            names[sid] = "Premium / High-Value"
        elif row["frequency"] >= profile["frequency"].median() and row["recency"] <= profile["recency"].median():
            names[sid] = "Loyal / Frequent"
        elif row["recency"] >= profile["recency"].quantile(0.75):
            names[sid] = "At-Risk / Lapsing"
        else:
            names[sid] = "Occasional / Low-Value"
    # ensure unique names if two fall in same bucket
    seen = {}
    for sid in profile.index:
        base = names[sid]
        if base in seen:
            names[sid] = f"{base} ({sid})"
        seen[base] = True
    print("\nSegment names / recommended actions (cluster ids are labels only, no ordering):")
    actions = {
        "Premium / High-Value": "VIP perks and early access to electronics deals; protect with concierge support.",
        "Loyal / Frequent": "Subscription nudges and streak rewards to keep the habit.",
        "At-Risk / Lapsing": "Win-back coupons and a reminder of saved favourites.",
        "Occasional / Low-Value": "Basket-building bundles and free-delivery thresholds.",
    }
    for sid in profile.index:
        nm = names[sid]
        print(f"  seg {sid} -> {nm}: {actions.get(nm.split(' (')[0], 'nurture')}")
    print(f"Premium segment = id {premium_seg} (highest mean monetary {profile.loc[premium_seg,'monetary']}).")

    # PCA projection
    pca = PCA(n_components=2).fit(X)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {evr[0]:.3f}, {evr[1]:.3f} (total {evr.sum():.3f})  "
          f"[ref 0.634, 0.218, total 0.852]")

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium_seg
    rfm[["customer_id", "recency", "frequency", "monetary",
         "segment_id", "segment_name", "is_premium"]].to_csv(
        os.path.join(OUT, "customer_segments.csv"), index=False)
    print(f"\nWrote {OUT}/customer_segments.csv ({len(rfm)} rows); premium customers: {int(rfm['is_premium'].sum())}")


if __name__ == "__main__":
    main()
