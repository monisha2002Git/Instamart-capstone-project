"""Part A, task 6 -- run the seven queries in sql/queries.sql and report their results."""
import os
import re
import sqlite3
import pandas as pd

DB = os.path.join("data", "instamart.db")
SQLF = os.path.join("sql", "queries.sql")


def load_queries():
    text = open(SQLF, encoding="utf-8").read()
    parts = re.split(r"--\s*name:\s*(\w+)\s*\n", text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    queries = {}
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i + 1]
        # strip trailing comment-only lines but keep the SQL
        queries[name] = body.strip()
    return queries


def main():
    conn = sqlite3.connect(DB)
    q = load_queries()

    q1 = pd.read_sql_query(q["Q1"], conn)
    print(f"Q1 (WHERE+HAVING): {len(q1)} customers  [ref 30]")

    q2 = pd.read_sql_query(q["Q2"], conn)
    print(f"Q2 (3-table join): {len(q2)} rows, sum INR {round(q2['revenue'].sum(),2)}  [ref 30 rows, 2998668.80]")
    top = q2.iloc[0]
    print(f"   largest cell: {top['city']} / {top['category']} = INR {top['revenue']}  [ref Bengaluru/Electronics 233909.45]")

    q3 = pd.read_sql_query(q["Q3"], conn)
    print(f"Q3 (scalar subquery): {len(q3)} products above avg price  [ref 11]")

    q4 = pd.read_sql_query(q["Q4"], conn)
    print(f"Q4 (LEFT JOIN): {len(q4)} customers with no completed order  [ref 25]")

    q5 = pd.read_sql_query(q["Q5"], conn)
    # recompute the average for reporting
    spend = pd.read_sql_query(
        "SELECT customer_id, SUM(order_amount) t FROM orders WHERE status='completed' GROUP BY customer_id", conn)
    print(f"Q5 (CTE): {len(q5)} customers above avg; avg INR {round(spend['t'].mean(),2)}  [ref 550, 2552.06]")

    q6 = pd.read_sql_query(q["Q6"], conn)
    tops = q6[q6["rnk"] == 1].sort_values("city")
    print("Q6 (window ranks): top customer per city ->")
    for _, r in tops.iterrows():
        print(f"   {r['city']}: {r['customer_id']} ({r['total_spend']})")
    n_rank_diff = int((q6["rnk"] != q6["row_num"]).sum())
    print(f"   rows where RANK() != ROW_NUMBER(): {n_rank_diff}  [ref 8]")

    q7 = pd.read_sql_query(q["Q7"], conn)
    print(f"Q7 (running total): {len(q7)} daily rows; last cumulative INR {q7['cumulative_revenue'].iloc[-1]}  "
          f"[ref 183, 2998668.80]")

    conn.close()


if __name__ == "__main__":
    main()
