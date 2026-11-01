"""Part C -- support-ticket triage with guardrails, redaction, retries, validation,
cost control and segment-aware routing. Reads data/raw/tickets.csv and
outputs/customer_segments.csv; writes outputs/triage_results.csv and
outputs/triage_budget_demo.csv. All model calls go through mock_llm.py."""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for mock_llm
import mock_llm
from mock_llm import est_tokens, TransientLLMError

RAW = os.path.join("data", "raw")
OUT = "outputs"
ALLOWED = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

# ---- task 2: redaction patterns, applied strictly in this order ----
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact(text):
    counts = {}
    for name, pat, repl in REDACTIONS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts


# ---- task 3: prompts ----
ZERO_SHOT = (
    "You CLASSIFY an Instamart support ticket into exactly one category from this list: "
    "billing, delivery, technical, fraud. "
    'Reply with JSON only in the form {"category": "...", "reason": "..."}.'
)
FEW_SHOT = (
    "You CLASSIFY an Instamart support ticket into exactly one category from this list: "
    "billing, delivery, technical, fraud. "
    'Reply with JSON only in the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Ticket: "An unauthorized transaction appeared on my card." -> {"category": "fraud", "reason": "unauthorized charge"}\n'
    'Ticket: "The rider never delivered my order." -> {"category": "delivery", "reason": "non-delivery"}\n'
    'Ticket: "I was charged twice and need a refund." -> {"category": "billing", "reason": "double charge"}'
)


# ---- task 4: resilient model call ----
def call_with_retry(messages, ticket_id, log):
    attempt = 0
    while True:
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            log["errors"] += 1
            if attempt >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"[retry] ticket={ticket_id} retry={attempt} wait={wait:.4f}s")
            time.sleep(wait)
            attempt += 1


# ---- task 5: output validation ----
def validate(text):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("category") in ALLOWED:
            return "classified_json", obj["category"]
    except (json.JSONDecodeError, TypeError):
        pass
    words = re.findall(r"[a-z]+", (text or "").lower())
    present = [c for c in ALLOWED if c in words]
    if len(present) == 1:
        return "classified_keyword", present[0]
    return "flagged_invalid_output", None


def run_pipeline(tickets, system_prompt, token_budget, seed=42):
    """Process tickets in file order; return (records, log)."""
    random.seed(seed)
    mock_llm.reset()
    log = {"errors": 0}
    total_tokens = 0
    records = []
    for _, t in tickets.iterrows():
        tid, cid, text = t["ticket_id"], t["customer_id"], t["text"]
        rec = {"ticket_id": tid, "customer_id": cid, "status": None,
               "category": None, "tokens_used": 0}

        if text is None or text.strip() == "":
            rec["status"] = "skipped_empty"
            records.append(rec); continue
        if est_tokens(text) > MAX_INPUT_TOKENS:
            rec["status"] = "flagged_long"
            records.append(rec); continue
        if total_tokens >= token_budget:
            rec["status"] = "budget_exhausted"
            records.append(rec); continue

        redacted, _ = redact(text)
        # privacy assertion: nothing sensitive leaves in the prompt
        for _, pat, _repl in REDACTIONS:
            assert not pat.search(redacted), f"PII leak in prompt for {tid}"
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": redacted}]
        resp = call_with_retry(messages, tid, log)
        total_tokens += resp.usage.total_tokens
        rec["tokens_used"] = resp.usage.total_tokens
        status, cat = validate(resp.choices[0].message.content)
        rec["status"] = status
        rec["category"] = cat
        records.append(rec)
    return records, log, total_tokens


def report_redaction(tickets):
    totals = {"email": 0, "phone": 0, "order": 0}
    redacted_texts = []
    for text in tickets["text"]:
        r, c = redact(text)
        for k in totals:
            totals[k] += c[k]
        redacted_texts.append(r)
    total = sum(totals.values())
    print(f"\nRedactions across {len(tickets)} tickets: "
          f"{totals['email']} emails, {totals['phone']} phones, {totals['order']} order ids, {total} total  "
          f"[ref 55 / 91 / 132 / 278]")
    leftover = 0
    for r in redacted_texts:
        for _, pat, _ in REDACTIONS:
            leftover += len(pat.findall(r))
    print(f"  re-applying the three patterns to redacted texts finds: {leftover}  [ref 0]")


def summarise(records):
    from collections import Counter
    return Counter(r["status"] for r in records)


def accuracy(records, truth):
    classified = [r for r in records if r["status"] in ("classified_json", "classified_keyword")]
    correct = sum(1 for r in classified if r["category"] == truth[r["ticket_id"]])
    return len(classified), correct, (100.0 * correct / len(classified) if classified else 0.0)


def confusion(records, truth):
    from collections import defaultdict
    table = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r["status"] in ("classified_json", "classified_keyword"):
            table[truth[r["ticket_id"]]][r["category"]] += 1
    return table


def route(records, segments):
    seg_name = dict(zip(segments["customer_id"], segments["segment_name"]))
    premium = set(segments.loc[segments["is_premium"], "customer_id"])
    rows = []
    human = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
    for r in records:
        status, cat, cid = r["status"], r["category"], r["customer_id"]
        if status in human:
            route_to, priority = "human_review", "NORMAL"
        elif cat == "fraud":
            route_to, priority = "security_team", "HIGH"
        elif cid in premium:
            route_to, priority = "premium_desk", "HIGH"
        else:
            route_to, priority = "support_queue", "NORMAL"
        rows.append({
            "ticket_id": r["ticket_id"], "customer_id": cid, "status": status,
            "category": cat if cat else "",
            "segment_name": seg_name.get(cid, ""),
            "priority": priority, "route": route_to,
            "tokens_used": r["tokens_used"],
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
    truth = dict(zip(tickets["ticket_id"], tickets["true_category"]))
    segments = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))

    # task 1 figures
    empty = tickets["text"].str.strip().eq("").sum()
    longt = tickets["text"].map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum()
    long_est = tickets.loc[tickets["text"].map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS), "text"].map(est_tokens)
    print(f"Input guardrails: {empty} empty, {longt} long (est {long_est.iloc[0] if longt else '-'} tokens)  "
          f"[ref 4 empty, 4 long @ 734]")

    report_redaction(tickets)

    # task 5/6: full zero-shot run (no budget)
    print("\n--- zero-shot full run ---")
    zrec, zlog, ztok = run_pipeline(tickets, ZERO_SHOT, token_budget=10 ** 12)
    zc = summarise(zrec)
    print(f"status counts: json={zc['classified_json']} keyword={zc['classified_keyword']} "
          f"invalid={zc['flagged_invalid_output']}  [ref 135 / 12 / 9]")
    print(f"TransientLLMErrors encountered (all recovered): {zlog['errors']}  [ref 19]")
    n_cls, n_ok, acc = accuracy(zrec, truth)
    print(f"classified {n_cls}, accuracy {acc:.1f}% ({n_ok}/{n_cls})  [ref 147, 85.7%]")
    print(f"total tokens {ztok}, tokens/classified {ztok/n_cls:.1f}")

    print("zero-shot confusion (true -> predicted):")
    conf = confusion(zrec, truth)
    for tc in ALLOWED + ["none"]:
        if tc in conf:
            print(f"  {tc}: {dict(conf[tc])}")

    # task 6: few-shot full run
    print("\n--- few-shot full run ---")
    frec, flog, ftok = run_pipeline(tickets, FEW_SHOT, token_budget=10 ** 12)
    fc = summarise(frec)
    fn_cls, fn_ok, facc = accuracy(frec, truth)
    print(f"status counts: json={fc['classified_json']} keyword={fc['classified_keyword']} "
          f"invalid={fc['flagged_invalid_output']}")
    print(f"classified {fn_cls}, accuracy {facc:.1f}% ({fn_ok}/{fn_cls})  [ref 100%]")
    print(f"total tokens {ftok}, tokens/classified {ftok/fn_cls:.1f}")

    # task 7a: budget demo (zero-shot, TOKEN_BUDGET=6000)
    BUDGET = 6000
    brec, blog, btok = run_pipeline(tickets, ZERO_SHOT, token_budget=BUDGET)
    bc = summarise(brec)
    df_budget = route(brec, segments)
    df_budget.to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
    print(f"\nbudget demo (TOKEN_BUDGET={BUDGET}): {bc['budget_exhausted']} tickets budget_exhausted "
          f"-> outputs/triage_budget_demo.csv")

    # task 7b: final results (few-shot, TOKEN_BUDGET=100000) with routing
    frec2, _, _ = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    df_final = route(frec2, segments)
    df_final.to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
    print(f"final routing -> outputs/triage_results.csv ({len(df_final)} rows)")
    print("route counts:", df_final["route"].value_counts().to_dict())
    print("priority counts:", df_final["priority"].value_counts().to_dict())


if __name__ == "__main__":
    main()
