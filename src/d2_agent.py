"""Part D, tasks 5-9 -- refund agent: tools, ReAct loop, stop-safely fallbacks,
resumable checkpoints. Runs the seven agent_scenarios and writes
outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and state/ checkpoints.
All model calls go through mock_llm.py; no network, no API key."""
import json
import os
import sqlite3
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import mock_llm
from d1_rag import Retriever

DB = os.path.join("data", "instamart.db")
RAW = os.path.join("data", "raw")
OUT = "outputs"
STATE = "state"
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")

AGENT_SYSTEM = (
    "You are a refund AGENT. Decide the refund for the customer's order by calling the provided tools "
    "one at a time, then reply with a final decision prefixed by REFUND, ESCALATE, DENIED or NOT_FOUND."
)

_RETRIEVER = Retriever(overlap=10)


class SimulatedCrash(Exception):
    """Injected to demonstrate checkpoint/resume."""


# ---------------- tools ----------------
def get_order(order_id):
    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)).fetchone()
    conn.close()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2],
            "delivery_minutes": (None if dm is None or dm == "" else float(dm))}


def search_policy(query):
    return {"results": _RETRIEVER.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(OUT, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "tool": tool, "order_id": order_id,
                            "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the policy chunks most relevant to a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund due given order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue a refund and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                     "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Escalate a high-value refund to a human lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                     "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}


# ---------------- checkpointing ----------------
def save_checkpoint(st):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def load_checkpoint(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------- ReAct driver ----------------
def _finish(st, status, final_answer):
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "tools_done": st["tools_done"]}


def _drive(st, max_iterations, token_limit, time_limit_seconds, start, crash_after=None):
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= time_limit_seconds:
            return _finish(st, "FALLBACK_TIMEOUT",
                           "Handing to a human agent: time limit reached before a decision.")
        if st["llm_calls"] >= max_iterations:
            return _finish(st, "FALLBACK_MAX_ITERATIONS",
                           "Handing to a human agent: could not resolve within the step budget.")

        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:
            return _finish(st, "FALLBACK_TOKEN_LIMIT",
                           "Handing to a human agent: token budget exceeded before a decision.")

        msg = resp.choices[0].message
        if not msg.tool_calls:
            status = (msg.content or "").split(":")[0].strip()
            return _finish(st, status, msg.content)

        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        observation = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(observation)})
        st["tools_done"].append(tc.name)
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        st["step"] += 1
        save_checkpoint(st)

        if crash_after is not None and len(st["tools_done"]) == crash_after:
            raise SimulatedCrash(f"crash after {crash_after} tools (checkpoint saved)")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after=None):
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_SYSTEM},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_done": [], "steps": []}
    return _drive(st, max_iterations, token_limit, time_limit_seconds,
                  start=time.monotonic(), crash_after=crash_after)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    st = load_checkpoint(run_id)
    st.setdefault("steps", [])
    st.setdefault("llm_calls", 0)
    return _drive(st, max_iterations, token_limit, time_limit_seconds, start=time.monotonic())


# ---------------- helpers ----------------
def refund_amount(result):
    """Refund actually decided: from issue_refund / escalate step, else 0."""
    for s in result["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return float(s["arguments"]["refund_inr"])
    return 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STATE, exist_ok=True)
    if os.path.exists(LEDGER):
        os.remove(LEDGER)

    scen = pd.read_csv(os.path.join(RAW, "agent_scenarios.csv"), keep_default_na=False)
    runs_path = os.path.join(OUT, "agent_runs.jsonl")
    if os.path.exists(runs_path):
        os.remove(runs_path)

    print("Seven scenario results:")
    print(f"{'id':4} {'status':24} {'refund':>9} {'calls':>6} {'tokens':>7}  expected")
    for _, s in scen.iterrows():
        res = run_agent(s["message"], s["scenario_id"], run_id=s["scenario_id"])
        amt = refund_amount(res)
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"scenario_id": res["scenario_id"], "status": res["status"],
                                "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                                "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
        print(f"{s['scenario_id']:4} {res['status']:24} {amt:9.2f} {res['llm_calls']:6} "
              f"{res['tokens_used']:7}  (exp {s['expected_outcome']} {s['expected_refund_inr']})")

    # ---- task 7: stop-safely demonstrations ----
    print("\nStop-safely demonstrations:")
    s1 = scen[scen["scenario_id"] == "S1"].iloc[0]
    tok = run_agent(s1["message"], "S1", token_limit=200, run_id="S1_tokendemo")
    print(f"  S1 token_limit=200 -> {tok['status']}  [ref FALLBACK_TOKEN_LIMIT]")
    tmo = run_agent(s1["message"], "S1", time_limit_seconds=0, run_id="S1_timeoutdemo")
    print(f"  S1 time_limit_seconds=0 -> {tmo['status']}  [ref FALLBACK_TIMEOUT]")

    # ---- task 9: crash-and-resume for S1 ----
    print("\nCrash-and-resume (S1):")
    tools_before = []
    try:
        run_agent(s1["message"], "S1", run_id="S1_resume", crash_after=2)
    except SimulatedCrash as e:
        ckpt = load_checkpoint("S1_resume")
        tools_before = list(ckpt["tools_done"])
        print(f"  crashed: {e}; tools so far: {tools_before}")
    resumed = resume_agent("S1_resume")
    print(f"  after resume: status {resumed['status']}, refund {refund_amount(resumed):.2f}")
    all_tools = resumed["tools_done"]
    print(f"  tools executed across crash+resume (in order): {all_tools}")
    assert all_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"], all_tools
    assert resumed["status"] == "REFUND" and abs(refund_amount(resumed) - 128.25) < 0.01
    print("  [ref get_order, search_policy, calculate_refund, issue_refund; REFUND 128.25]")


if __name__ == "__main__":
    main()
