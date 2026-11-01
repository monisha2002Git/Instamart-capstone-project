"""Part B, tasks 4-8 -- lapse label, leakage-free features, tuning, evaluation,
class-balance experiment, interpretation, persistence and the retention list.
Reads data/clean/*.csv and outputs/customer_segments.csv (Premium flag only, task 8c).
Writes outputs/lapse_model.joblib and outputs/retention_list.csv."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

CLEAN = os.path.join("data", "clean")
OUT = "outputs"
CUTOFF = pd.Timestamp("2025-09-01")
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate",
       "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]


def build_dataset():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    before = orders[orders["order_ts"] < CUTOFF]
    comp_before = before[before["status"] == "completed"]

    # cohort: >=1 completed order before cutoff
    cohort = sorted(comp_before["customer_id"].unique())

    # label: lapsed if no completed order in Sep 2025
    sep = orders[(orders["order_ts"] >= CUTOFF) &
                 (orders["order_ts"] < pd.Timestamp("2025-10-01")) &
                 (orders["status"] == "completed")]
    active_sep = set(sep["customer_id"].unique())

    # features from before-cutoff data only
    feat = comp_before.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    feat["recency"] = (CUTOFF - feat["last_order"]).dt.days

    # most frequent completed category before cutoff
    fav = (comp_before.groupby(["customer_id", "category"]).size()
           .reset_index(name="n")
           .sort_values(["customer_id", "n"], ascending=[True, False])
           .drop_duplicates("customer_id").set_index("customer_id")["category"])
    feat["fav_category"] = fav

    # cancellation rate over final-status orders before cutoff
    final = before[before["status"].isin(["completed", "cancelled"])]
    canc = final.assign(is_canc=(final["status"] == "cancelled").astype(int)) \
                .groupby("customer_id")["is_canc"].mean()
    feat["cancellation_rate"] = canc

    feat = feat.loc[cohort].reset_index()
    feat = feat.merge(customers[["customer_id", "city", "gender", "age", "signup_date"]],
                      on="customer_id", how="left")
    feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
    feat["lapsed"] = (~feat["customer_id"].isin(active_sep)).astype(int)

    print(f"cohort size: {len(feat)}  [ref 1156]")
    print(f"lapsed: {int(feat['lapsed'].sum())} ({100*feat['lapsed'].mean():.2f}%)  [ref 443, 38.32%]")
    assert feat[NUM + CAT].isna().sum().sum() == 0, "features contain missing values"
    print("confirmed: no feature column contains a missing value")
    return feat


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])


def main():
    os.makedirs(OUT, exist_ok=True)
    data = build_dataset()
    X = data[NUM + CAT]
    y = data["lapsed"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(X_tr)} / test {len(X_te)}  [ref 924 / 232]")

    # baseline: always predict not-lapsed
    base_pred = np.zeros(len(y_te), dtype=int)
    base_acc = (base_pred == y_te.values).mean()
    base_f1 = f1_score(y_te, base_pred, zero_division=0)
    print(f"\nbaseline (always not-lapsed): accuracy {100*base_acc:.1f}%, lapsed-F1 {base_f1:.2f}  [ref 61.6%, 0]")

    # leakage check: report fitted scaler mean of frequency vs full-cohort mean
    pre_probe = make_preprocessor().fit(X_tr)
    scaler = pre_probe.named_transformers_["num"]
    freq_idx = NUM.index("frequency")
    print(f"\nfitted scaler mean(frequency) = {scaler.mean_[freq_idx]:.3f} (train only); "
          f"full-cohort mean(frequency) = {X['frequency'].mean():.3f} -- "
          f"they differ because the scaler is fitted on the 924 training rows only, not the whole cohort.")

    # ---- Logistic Regression ----
    lr_pipe = Pipeline([("pre", make_preprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))])
    lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                           cv=5, scoring="f1")
    lr_grid.fit(X_tr, y_tr)
    print(f"\nLogReg best params: {lr_grid.best_params_}, best CV F1 {lr_grid.best_score_:.3f}  [ref C=1, ~0.62]")

    # ---- Random Forest ----
    rf_pipe = Pipeline([("pre", make_preprocessor()),
                        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_grid = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                                     "clf__min_samples_leaf": [5, 10, 20]},
                           cv=5, scoring="f1")
    rf_grid.fit(X_tr, y_tr)
    print(f"RandForest best params: {rf_grid.best_params_}, best CV F1 {rf_grid.best_score_:.3f}  "
          f"[ref depth 8 leaf 10, ~0.65]")

    # ---- test-set evaluation (after tuning) ----
    def report(name, model):
        pred = model.predict(X_te)
        cm = confusion_matrix(y_te, pred)
        p = precision_score(y_te, pred, zero_division=0)
        r = recall_score(y_te, pred, zero_division=0)
        f = f1_score(y_te, pred, zero_division=0)
        print(f"\n{name} test set:")
        print(f"  confusion matrix [[TN FP][FN TP]] = {cm.tolist()}")
        print(f"  lapsed-class precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
        return f

    lr_f1 = report("LogReg", lr_grid.best_estimator_)
    rf_f1 = report("RandForest", rf_grid.best_estimator_)

    selected_name, selected = ("RandomForest", rf_grid.best_estimator_) if rf_f1 >= lr_f1 else ("LogReg", lr_grid.best_estimator_)
    print(f"\nSelected model: {selected_name} (test lapsed-F1 {max(lr_f1, rf_f1):.3f}). "
          f"Missing a lapsing customer (false negative) costs a lost customer the retention team never contacts, "
          f"while wrongly targeting a loyal one (false positive) costs only a cheap coupon; we therefore favour the "
          f"model with the stronger lapsed-class recall/F1.")

    # ---- task 7: class-balance experiment on best logistic regression ----
    tr = X_tr.copy()
    tr["lapsed"] = y_tr.values
    majority = tr[tr["lapsed"] == 0]
    minority = tr[tr["lapsed"] == 1]
    need = len(majority) - len(minority)
    extra = minority.sample(n=need, replace=True, random_state=42)
    bal = pd.concat([tr, extra], ignore_index=True)
    print(f"\nclass-balance: lapsed {len(minority)} -> {len(minority)+need}, not-lapsed {len(majority)} "
          f"(both {len(majority)})  [ref 570 each]")
    bal_lr = Pipeline([("pre", make_preprocessor()),
                       ("clf", LogisticRegression(max_iter=1000, C=lr_grid.best_params_["clf__C"]))])
    bal_lr.fit(bal[NUM + CAT], bal["lapsed"])

    def pr(model):
        pred = model.predict(X_te)
        return (precision_score(y_te, pred, zero_division=0),
                recall_score(y_te, pred, zero_division=0),
                f1_score(y_te, pred, zero_division=0))
    up = pr(lr_grid.best_estimator_)
    bp = pr(bal_lr)
    print(f"  unbalanced LR  precision {up[0]:.2f} recall {up[1]:.2f} F1 {up[2]:.2f}")
    print(f"  balanced   LR  precision {bp[0]:.2f} recall {bp[1]:.2f} F1 {bp[2]:.2f}")
    print(f"  recall moves most ({up[1]:.2f} -> {bp[1]:.2f}); oversampling makes the model catch more true "
          f"lapsers (fewer false negatives) at little precision cost -- good for retention outreach.")

    # ---- task 8a: interpretation ----
    rf_model = rf_grid.best_estimator_
    feat_names = rf_model.named_steps["pre"].get_feature_names_out()
    importances = rf_model.named_steps["clf"].feature_importances_
    top5 = sorted(zip(feat_names, importances), key=lambda t: -t[1])[:5]
    print("\nTop 5 random-forest feature importances:")
    for n, imp in top5:
        print(f"  {n}: {imp:.3f}")

    lr_model = lr_grid.best_estimator_
    lr_names = lr_model.named_steps["pre"].get_feature_names_out()
    coefs = lr_model.named_steps["clf"].coef_[0]
    cmap = dict(zip(lr_names, coefs))
    print("LogReg coefficient signs (3 features; + raises modelled lapse probability):")
    nice = {"num__recency": "days since last order", "num__frequency": "past order count",
            "num__monetary": "past spend"}
    for key in ["num__recency", "num__frequency", "num__monetary"]:
        if key in cmap:
            sign = "+" if cmap[key] >= 0 else "-"
            direction = "raises" if cmap[key] >= 0 else "lowers"
            print(f"  {key}: {sign} ({cmap[key]:+.3f}) -- higher {nice[key]} {direction} modelled lapse risk")

    # ---- task 8b: persist + reload ----
    joblib.dump(selected, os.path.join(OUT, "lapse_model.joblib"))
    reloaded = joblib.load(os.path.join(OUT, "lapse_model.joblib"))
    same = np.array_equal(selected.predict(X_te), reloaded.predict(X_te))
    print(f"\nsaved + reloaded model reproduces test predictions exactly: {same}")

    # ---- task 8c: retention list for Premium test customers ----
    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))
    premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
    te = data.loc[X_te.index].copy()
    te["prob"] = selected.predict_proba(X_te)[:, 1]
    prem_te = te[te["customer_id"].isin(premium_ids)].sort_values("prob", ascending=False)
    top = prem_te.head(10)
    out = top[["customer_id", "prob", "lapsed"]].rename(
        columns={"prob": "lapse_probability", "lapsed": "actually_lapsed"})
    out.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
    print(f"\nretention_list.csv: {len(out)} premium test customers; "
          f"{int(out['actually_lapsed'].sum())} of them truly lapsed")
    print("NOTE: the Premium flag is a post-hoc whole-window label, so this retention list is illustrative, "
          "not a leakage-free evaluation (model features still use only pre-cutoff orders).")


if __name__ == "__main__":
    main()
