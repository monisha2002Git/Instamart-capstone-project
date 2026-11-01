# Instamart Customer Intelligence & Support Copilot

One internal tool, four connected Parts, built on a single synthetic dataset:
**Part A** turns messy exports into a trusted SQLite database; **Part B** segments
customers and predicts lapse; **Part C** triages support tickets with privacy and
cost guardrails; **Part D** answers policy questions and runs a tool-calling refund
agent. All data is synthetic and fictional. All monetary values are in **INR** — no
figure uses a dollar sign. Every model call goes through the offline `mock_llm.py`
(no API key, no network, no cost).

## Overview

| Part | What it produces | Key artifacts |
|------|------------------|---------------|
| A | Clean tables + database + SQL analysis | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | Segments + lapse model | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Guardrailed ticket triage | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG + refund agent | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Source lives in `src/` (`a1`–`a5`, `b1`–`b2`, `c1`, `d1`–`d2`). The two supplied
files `generate_data.py` and `mock_llm.py` are committed **unmodified**.

## Reproduce

Run from the repository root, in order. A grader can paste these verbatim.

```bash
# 1. environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. generate the synthetic raw data (prints: generated: customers=1200 products=40 orders_raw=13084 tickets=164)
python generate_data.py

# 3. Part A -- audit, clean, store, validate, SQL, EDA
python src/a1_audit.py
python src/a2_clean.py
python src/a3_validate.py            # prints ALL CHECKPOINTS PASSED
python src/a4_sql.py
python src/a5_pandas.py

# 4. Part B -- segments then lapse model (b2 reads outputs/customer_segments.csv)
python src/b1_segments.py
python src/b2_lapse.py

# 5. Part C -- support-ticket triage (reads outputs/customer_segments.csv)
python src/c1_triage.py

# 6. Part D -- policy RAG, then the refund agent (reads data/instamart.db + data/policies/)
python src/d1_rag.py
python src/d2_agent.py
```

Every randomised step uses the fixed seeds from the brief (`random.Random(42)` in the
data generator, `random_state=42` in all scikit-learn estimators and the oversampler,
`random.seed(42)` for the retry jitter).

---

## Part A — Data Foundation

### Audit of the raw exports (task 2)

| Figure | Value | Reference |
|---|---|---|
| orders.csv rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status `pending` | 656 | 656 |
| distinct `order_id` | 12,309 | 12,309 |
| non-numeric `order_amount` | 120 | 120 |
| negative `delivery_minutes` | 106 | 106 |
| blank `delivery_minutes` | 2,541 | 2,541 |
| blank `age` | 61 | 61 |
| `age` outside 10–100 | 6 | 6 |
| distinct raw city spellings | 24 | 24 |
| distinct non-blank gender spellings | 10 | 10 |
| blank `gender` | 30 | 30 |

`tickets.csv` is always read with `keep_default_na=False` so empty ticket texts stay
empty strings.

### Cleaning rules and rows changed

**Customers (task 3)**

| Rule | Effect |
|---|---|
| City: strip whitespace + title case | 24 raw spellings → 6 cities |
| Gender → male/female/unknown | 30 blanks → `unknown`; all variants (M/m/MALE…) folded |
| Age: out-of-range → missing, then fill with city median (`groupby.transform`) | 61 blank + 6 out-of-range = 67 ages filled; 0 missing after |
| `signup_date` → datetime | type conversion |

Result: cities **Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191,
Mumbai 184**; genders **female 610, male 560, unknown 30**; **0 missing ages**, all in 18–60.

**Orders (task 4), applied in this order**

| # | Rule | Rows/values changed |
|---|---|---|
| 1 | `order_ts` → datetime | type conversion |
| 2 | Drop exact duplicate rows | 13,084 → 12,965 (−119) |
| 3 | Keep latest `order_ts` per `order_id` | 12,965 → 12,309 (−656 earlier pending records) |
| 4 | `order_amount` → numeric; unparseable → recompute `price×qty×(1−disc/100)` | 112 values recomputed; 0 missing after |
| 5 | Negative `delivery_minutes` → missing; completed filled with category median; cancelled left missing | 105 negatives blanked; 774 missing-among-completed filled; 1,199 cancelled left missing |

Result: **12,309 rows** (one per `order_id`), **11,110 completed / 1,199 cancelled**,
no pending; **0 missing `order_amount`**; **0 missing delivery among completed**,
**1,199 missing among cancelled**; completed `order_amount` sum = **INR 2,998,668.80**.

> Note: the raw-audit counts (120 non-numeric, 106 negative) fall slightly after
> de-duplication/collapse (112 recomputed, 105 blanked) because some dirty values
> sat on rows that were exact duplicates or superseded pending records.

### Validation (task 5)

`python src/a3_validate.py` asserts **19** reference checkpoints from tasks 3–4 (far
more than the required eight) and the presence of the three database tables, then
prints exactly `ALL CHECKPOINTS PASSED`.

### SQL analysis (task 6) — `sql/queries.sql`

| Query | Technique | Result | Reference |
|---|---|---|---|
| Q1 | WHERE then HAVING | 30 customers | 30 |
| Q2 | 3-table join | 30 rows, sum INR 2,998,668.80, top = Bengaluru/Electronics **INR 233,909.45** | matches |
| Q3 | scalar subquery | 11 products above average catalogue price | 11 |
| Q4 | LEFT JOIN | 25 customers with no completed order | 25 |
| Q5 | CTE + CTE | 550 customers above the average total; average **INR 2,552.06** | 550 / 2,552.06 |
| Q6 | RANK/DENSE_RANK/ROW_NUMBER over `PARTITION BY city` | top per city below; **8** rows have `RANK() ≠ ROW_NUMBER()` | 8 |
| Q7 | running total | 183 daily rows; last cumulative **INR 2,998,668.80** | 183 |

Q6 top customer per city: Bengaluru **C0082** (12,856.20), Chennai **C0930** (10,151.40),
Delhi **C0291** (13,742.60), Hyderabad **C0743** (12,730.05), Mumbai **C0693** (14,766.55),
Pune **C0426** (10,377.10).

*Why 8 rows differ:* when two customers in the same city have equal `total_spend`,
`RANK()` gives them the same rank and then skips the next value, while `ROW_NUMBER()`
assigns distinct consecutive numbers — so tied rows get different values from the two
functions.

### pandas cross-check and EDA (task 7)

`pivot_table(..., margins=True)` grand total = **INR 2,998,668.80**, equal to the SQL
total. Per-city completed `order_amount` (sum / mean / count):

| City | sum | mean | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Distribution of completed `order_amount`: mean **269.91**, median **135.00**
(right-skewed), skewness **4.61**, Q1 **64.80**, Q3 **297.00**, upper fence **645.30**
with **897** outliers above and **0** below.

**Four questions.** *What is this data about?* Minutes-delivery grocery/essentials
orders across six Indian cities, one row per order with amount, discount, quantity and
delivery time. *What do the distributions look like?* Order value is strongly
right-skewed (mean ≫ median, skew ≈ 4.61): most baskets are small groceries while a
long tail of electronics drives revenue; 897 high-value orders sit beyond the upper
IQR fence. *Outliers/anomalies?* The 897 upper outliers are genuine big-ticket orders,
not errors; the planted dirty values (negative delivery, unparseable amounts) were
fixed in Part A. *Relationships?* Over the four numeric fields the largest correlation
in absolute value is **`order_amount` vs `discount_pct` ≈ −0.095** (next is
`order_amount` vs `quantity` ≈ +0.087). These are very weak, and **a correlation of
this size does not establish causation**.

---

## Part B — Customer Segments and Lapse Prediction

### RFM (task 1)
Snapshot **2025-10-01 00:00:00**, completed orders only: **1,175** customers; recency
median **19** (min 0, max 180); frequency mean **9.46**, max **33**; monetary mean
**INR 2,552.06**, max **INR 14,766.55**.

### K-means (task 2)

| K | inertia | silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Chosen K = 4**: silhouette peaks there and four groups are business-actionable.
Inertia at K=8 is always below K=2 because adding centroids can only move them closer
to points (at K = n points, inertia = 0); it measures within-sample fit, not
generalisation, so inertia alone cannot pick K — we read the elbow and the silhouette.

### Profiles, names, PCA (task 3)

| segment_id | size | mean recency | mean frequency | mean monetary | name | action |
|---|---|---|---|---|---|---|
| 0 | 436 | 28.9 | 3.8 | 1,057.62 | Occasional / Low-Value | basket-building bundles, free-delivery thresholds |
| 1 | 138 | 24.1 | 6.8 | **7,129.70** | **Premium / High-Value** | VIP perks, early electronics access, concierge support |
| 2 | 419 | 8.8 | 19.4 | 3,410.02 | Loyal / Frequent | subscription nudges, streak rewards |
| 3 | 182 | 109.0 | 2.2 | 685.99 | At-Risk / Lapsing | win-back coupons, saved-favourites reminder |

Cluster ids are **labels only, with no ordering**. **Premium = segment 1** (highest
mean monetary). PCA(2) `explained_variance_ratio_` = **0.634, 0.218** (total **0.852**).
Written to `outputs/customer_segments.csv` (`customer_id, recency, frequency, monetary,
segment_id, segment_name, is_premium`); 138 premium customers.

### Lapse problem (task 4)
Cutoff **2025-09-01**. Cohort = customers with ≥1 completed order before the cutoff =
**1,156**; **443 lapsed (38.32%)**. Split (stratified, `random_state=42`): **924 train
/ 232 test**. Baseline "always not-lapsed": accuracy **61.6%**, lapsed-class F1 **0**.
Features use **only pre-cutoff orders** plus the customer table (recency, frequency,
monetary, avg discount, cancellation rate, avg delivery, favourite category, city, age,
gender, tenure). No September data and **no segment labels** are used as features.

### No-leakage preprocessing (task 5)
`OneHotEncoder(drop="first")` on city/gender/favourite-category and `StandardScaler` on
numerics, both inside a `Pipeline` **fitted on training rows only**. Fitted scaler
`mean(frequency)` = **8.115** vs full-cohort mean = **8.029** — they differ because the
scaler sees only the 924 training rows, not the whole cohort. Code asserts **no feature
column contains a missing value**.

### Tuning and evaluation (task 6)

| Model | grid | best params | best CV F1 | test confusion `[[TN,FP],[FN,TP]]` | test precision / recall / F1 (lapsed) |
|---|---|---|---|---|---|
| Logistic Regression | C ∈ {0.01,0.1,1,10,100}, cv=5 | C=100 | 0.651 | `[[111,32],[31,58]]` | 0.644 / 0.652 / **0.648** |
| Random Forest | depth ∈ {3,5,8} × leaf ∈ {5,10,20}, cv=5 (45 fits) | depth=8, leaf=10 | 0.681 | `[[113,30],[31,58]]` | 0.659 / 0.652 / **0.655** |

(Numbers differ slightly from the brief's indicative figures because extra features —
cancellation rate, avg delivery, tenure — were added, as the brief permits; both are of
the stated order and the selected model clears test F1 ≥ 0.55.) **Selected: Random
Forest** (test lapsed-F1 0.655). A false negative means a lapsing customer is never
contacted and is lost; a false positive means a loyal customer gets an unneeded coupon.
Losing a customer costs far more than a coupon, so we favour the model with the stronger
lapsed-class recall/F1 — the random forest.

### Class-balance experiment (task 7)
Oversampled the **training set only** (duplicates of lapsed rows drawn with replacement,
`random_state=42`) to **570 / 570**, refit the best logistic regression, evaluated on the
**unchanged** test set:

| LR variant | precision | recall | F1 |
|---|---|---|---|
| unbalanced | 0.64 | 0.65 | 0.65 |
| balanced | 0.59 | **0.85** | 0.70 |

**Recall moved most** (0.65 → 0.85) while precision stayed near 0.6 — oversampling makes
the model catch far more true lapsers (fewer false negatives), which is exactly what a
retention campaign wants.

### Interpret, persist, use (task 8)
Top-5 random-forest importances: `frequency` 0.280, `monetary` 0.271, `recency` 0.115,
`avg_delivery` 0.061, `cancellation_rate` 0.052 — recent purchase behaviour dominates
lapse risk. Logistic-regression coefficient signs: `recency` **+0.040** (longer gap →
higher lapse risk), `frequency` **−1.698** (more past orders → lower risk), `monetary`
**+0.285** (weak, higher past spend slightly raises modelled risk once frequency is
accounted for). The selected model is saved to `outputs/lapse_model.joblib`, reloaded,
and **reproduces the test predictions exactly**. Among Premium test-set customers the top
10 by predicted lapse probability go to `outputs/retention_list.csv`
(`customer_id, lapse_probability, actually_lapsed`); **2 of 10 truly lapsed**.

> The Premium flag comes from the whole-window RFM (which includes September), so it is a
> **post-hoc whole-window label**: this retention list is **illustrative, not a
> leakage-free evaluation**. The model's own features still use only pre-cutoff orders.

**Leakage statement:** no encoder, scaler or resampler ever saw test rows; no data dated
2025-09-01 or later entered any feature; the segment labels from task 3 were not used as
features.

---

## Part C — Support Ticket Triage with Guardrails

Tickets are processed in file order. `MAX_INPUT_TOKENS = 400`.

- **Input guardrails (task 1):** 4 empty → `skipped_empty`, 4 long (**734** est. tokens
  each) → `flagged_long`; neither triggers a model call.
- **Redaction (task 2):** applied strictly email → phone → order-id. Across all 164
  tickets: **55 emails, 91 phones, 132 order ids, 278 total**; re-applying the three
  patterns to the redacted texts finds **0**. An assertion guarantees no email/phone/
  order-id reaches any prompt; the user message sent is exactly the redacted text.
- **Prompts (task 3):** a zero-shot and a few-shot system prompt, both containing
  `CLASSIFY`, the four categories and a JSON-only contract; the few-shot one adds an
  `EXAMPLES:` block with a fraud, delivery and billing example.
- **Resilient calls (task 4):** up to 3 retries, backoff `0.05 × 2**n × U(0.9,1.15)`;
  `mock_llm.reset()` before each run. A full run hits **19 TransientLLMErrors, all
  recovered** (every retry is retry 0, so each wait ∈ [0.045, 0.0575] s). No ticket ends
  in a permanent error.
- **Validation (task 5):** one full run → **135 `classified_json`, 12
  `classified_keyword`, 9 `flagged_invalid_output`**. Keyword extraction uses
  `re.findall(r"[a-z]+", …)` so `billing-related` splits into words; valid JSON with a
  category outside the list (e.g. `shipping`) is flagged, never guessed.

### Zero-shot vs few-shot (task 6)

| Run | classified | accuracy | total tokens | tokens / classified |
|---|---|---|---|---|
| zero-shot | 147 | **85.7%** (126/147) | 10,180 | 69.3 |
| few-shot | 147 | **100.0%** (147/147) | 18,604 | 126.6 |

Zero-shot confusion (true → predicted): billing→billing 38; delivery→delivery 36;
technical→{technical 29, billing 10}; fraud→{fraud 23, billing 11}. **Every error is a
fraud or technical ticket mislabelled billing**, exactly the reference pattern. The
few-shot prompt costs about **+57 tokens per ticket** (126.6 vs 69.3) and removes all 21
errors, lifting accuracy from 85.7% to 100%. For this workload — where a misrouted fraud
ticket is expensive — the extra tokens are clearly worth paying; if the categories were
easy and errors cheap, the leaner zero-shot prompt would be the better trade.

### Budget, routing, output (task 7)
- **Budget demo:** zero-shot with `TOKEN_BUDGET = 6000` → **64 tickets `budget_exhausted`**
  once the running token total reaches the cap (saved to `outputs/triage_budget_demo.csv`).
- **Final results:** few-shot with `TOKEN_BUDGET = 100000` → `outputs/triage_results.csv`,
  **164 rows**. Routing (reads `outputs/customer_segments.csv`): `human_review` 17
  (4 empty + 4 long + 9 invalid), `security_team` 34 (fraud, HIGH), `premium_desk` 11
  (premium, HIGH), `support_queue` 102 (NORMAL). Priority: HIGH 45, NORMAL 119.

**Which outputs are rule-validated vs sent to humans:** the 147 rows with status
`classified_json`/`classified_keyword` are machine-routed to `security_team`,
`premium_desk` or `support_queue`; the 17 guardrail/validation failures
(`skipped_empty`, `flagged_long`, `flagged_invalid_output`, `budget_exhausted`) are all
routed to `human_review` and never auto-classified.

---

## Part D — Policy RAG Assistant and Refund Agent

### Retrieval and grounding (tasks 1–4)
- **Chunks:** 40 words, overlap 10 → **15 chunks** (refund 4, delivery 3, payment 3,
  privacy 2, escalation 3).
- **Retrieval:** `TfidfVectorizer` + `cosine_similarity`; for `"refund policy late
  delivery"` the **top-3 chunks are all `refund_policy.txt`**. *TF* measures how often a
  term appears in a chunk; *IDF* down-weights terms common across chunks and up-weights
  rare, distinctive ones. *Limitation:* TF-IDF matches surface words, so two phrasings
  with the same meaning but different words score as unrelated.
- **Grounding eval** (`policy_eval.csv`, 12 answerable + 2 not covered):

| run | expected_doc in top-3 | answer contains expected_fact | unanswerable → fallback |
|---|---|---|---|
| overlap = 10 | 12/12 | 12/12 | 2/2 |
| overlap = 0 | 12/12 | **10/12** | 2/2 |

  With **zero overlap**, *"What delivery time makes a completed order eligible for a full
  refund?"* breaks: the retrieved chunk returns *"…receives a full refund of the order
  amount."* but the qualifying fact *"more than 45 minutes"* fell into the previous chunk,
  so the answer no longer contains `expected_fact`. (*"How soon is money returned when a
  payment fails but is deducted?"* breaks the same way.) The 10-word overlap exists to
  keep facts that straddle a boundary intact.

### Refund agent — seven scenarios (tasks 5–6)

| scenario | status | refund (INR) | model calls | tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,057 |
| S2 | REFUND | 135.00 | 5 | 1,057 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,044 |
| S4 | DENIED | 0.00 | 4 | 733 |
| S5 | DENIED | 0.00 | 3 | 433 |
| S6 | NOT_FOUND | 0.00 | 2 | 136 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,192 |

Each refund equals `expected_refund_inr` within 0.01; S1 uses 5 model calls and S7
finishes all six calls under the default `token_limit=6000`. Five tools are defined with
a `TOOLS_SPEC` list and a `TOOL_MAP` name→function dictionary: `get_order` (reads
`data/instamart.db`), `search_policy` (reuses the Part-D retriever), `calculate_refund`
(100% refund if delivered > 45 min, else 0), `issue_refund` and `escalate_to_human`
(each append a timestamped line to `outputs/refund_ledger.jsonl`). Every run appends one
line to `outputs/agent_runs.jsonl` with status, final answer, llm_calls, tokens and all
steps.

### Stop-safely demonstrations (task 7)

| demonstration | status |
|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** (stops without executing the requested tool) |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** (stops before the first model call) |

Each fallback returns the steps completed so far plus a short human-handoff summary.

### Crash-and-resume (task 9)
S1 raises a simulated crash right after the **second** tool's checkpoint is saved
(`state/S1_resume.json` holds `get_order`, `search_policy`); `resume_agent` reloads it and
continues. Tools executed **across the crash and the resume, in order**:

```
get_order → search_policy → calculate_refund → issue_refund
```

Exactly four executions, none repeated; final status **REFUND, INR 128.25**. Each
checkpoint stores the run id, scenario id, full message list, tokens used, step number and
the list of tools already executed.

### ReAct vs plan-and-execute (task 10)
This agent is **ReAct**: it calls the model once per step and lets each observation drive
the next tool. Measured cost ranges from **2 model calls** (S6, order not found) and **3**
(S5, cancelled) up to **5** (S1, a full refund path), because the loop stops as soon as the
decision is reached. A **plan-and-execute** design asks the model once for the whole step
list, then executes it, replanning only on error — so a clean refund could cost as little
as **2 model calls** (one plan + one confirmation) and fewer tokens on the happy path. The
difference shows on **unexpected observations**: ReAct handles a `NOT_FOUND` or a
cancelled order naturally — the next model call simply sees the surprising tool result and
switches to `DENIED`/`NOT_FOUND` — whereas a pre-made plan assumed a refund and must detect
the mismatch and trigger a replan, adding calls and code. For this refund workflow the tool
sequence is short, branchy (not-found / cancelled / within-45-min / over-1000 all diverge
early) and safety-critical, so I would **choose ReAct**: its modest extra calls buy robust,
observation-driven branching and a clean per-step audit trail, which matter more here than
shaving one or two model calls on the happy path.

No real network call or API key is used anywhere in Part D.

---

## Limitations and Next Steps

- **Lapse retention list is illustrative, not a clean evaluation.** The Premium flag is a
  **post-hoc whole-window label** (its RFM includes September, the label month), so
  `outputs/retention_list.csv` demonstrates scoring but is not a leakage-free metric. Next:
  recompute Premium from pre-cutoff data only for an honest evaluation.
- **TF-IDF retrieval is lexical.** It misses paraphrases with no shared words; a dense/
  embedding retriever (or a hybrid) would be more robust, and chunk size/overlap should be
  tuned per document rather than fixed at 40/10.
- **The classifier and agent are driven by a deterministic mock.** Accuracy and token
  counts reflect `mock_llm.py`, not a hosted model; a real deployment needs live evaluation,
  prompt-injection hardening and human review sampling.
- **Modelling scope.** The lapse model uses a single cutoff and a 30-day label window; a
  rolling-cutoff, time-series cross-validated setup with calibration would give more reliable
  probabilities, and segment stability should be checked over time.
- **Data caveats.** All figures come from one synthetic seed; correlations are weak and do
  not establish causation, and real Instamart data would carry seasonality and city effects
  not present here.
