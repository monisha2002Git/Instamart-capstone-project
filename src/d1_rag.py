"""Part D, tasks 1-4 -- policy retrieval (chunking + TF-IDF) and a grounded answerer.
Reused by the refund agent (src/d2_agent.py). All model calls go through mock_llm.py."""
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for mock_llm
import mock_llm

POL = os.path.join("data", "policies")
RAW = os.path.join("data", "raw")
DOC_ORDER = ["refund_policy.txt", "delivery_policy.txt", "payment_policy.txt",
             "privacy_policy.txt", "escalation_policy.txt"]

POLICY_SYSTEM = (
    "You are a POLICY ASSISTANT. Answer the question using ONLY the supplied context. "
    "If the answer is not in the context, say you could not find the information."
)


def chunk_words(words, size=40, overlap=10):
    """Chunks of `size` words stepping by (size - overlap); stop after the first
    chunk whose end reaches or passes the last word."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        end = start + size
        chunks.append(words[start:end])
        if end >= len(words):
            break
        start += step
    return chunks


def build_chunks(overlap=10):
    chunks = []
    for fn in DOC_ORDER:
        words = open(os.path.join(POL, fn), encoding="utf-8").read().split()
        for w in chunk_words(words, 40, overlap):
            chunks.append({"source": fn, "text": " ".join(w)})
    return chunks


class Retriever:
    def __init__(self, overlap=10):
        self.chunks = build_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def retrieve(self, query, k=3):
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]),
                 "text": self.chunks[i]["text"]} for i in order]


def answer(question, retriever):
    hits = retriever.retrieve(question, k=3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": POLICY_SYSTEM},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]


def evaluate(overlap, label):
    retriever = Retriever(overlap=overlap)
    qa = pd.read_csv(os.path.join(RAW, "policy_eval.csv"), keep_default_na=False)
    answerable = qa[qa["expected_doc"] != "NONE"]
    unanswerable = qa[qa["expected_doc"] == "NONE"]

    doc_hits = fact_hits = 0
    failures = []
    for _, row in answerable.iterrows():
        text, sources = answer(row["question"], retriever)
        if row["expected_doc"] in sources:
            doc_hits += 1
        else:
            failures.append((row["question"], "expected_doc not retrieved"))
        if row["expected_fact"] in text:  # case-sensitive substring
            fact_hits += 1
        else:
            failures.append((row["question"], f"fact missing (got: {text[:60]!r})"))

    fallback_hits = 0
    for _, row in unanswerable.iterrows():
        text, _ = answer(row["question"], retriever)
        if text.startswith("I could not find"):
            fallback_hits += 1

    print(f"\n[{label}] answerable={len(answerable)} unanswerable={len(unanswerable)}")
    print(f"  expected_doc in top-3: {doc_hits}/{len(answerable)}")
    print(f"  answer contains expected_fact: {fact_hits}/{len(answerable)}")
    print(f"  unanswerable -> fallback: {fallback_hits}/{len(unanswerable)}")
    if failures:
        print("  failures:")
        for q, why in failures:
            print(f"    - {q}  [{why}]")
    return doc_hits, fact_hits, fallback_hits, failures


def main():
    # task 1 chunk counts
    from collections import Counter
    chunks = build_chunks(overlap=10)
    counts = Counter(c["source"] for c in chunks)
    print(f"Total chunks: {len(chunks)}  [ref 15]")
    for fn in DOC_ORDER:
        print(f"  {fn}: {counts[fn]}")
    print("  [ref refund 4, delivery 3, payment 3, privacy 2, escalation 3]")

    # task 2 retrieval demo
    r = Retriever(overlap=10)
    hits = r.retrieve("refund policy late delivery", k=3)
    print("\nretrieve('refund policy late delivery') top-3 sources:",
          [h["source"] for h in hits], " [ref all refund_policy.txt]")
    print("TF measures how often a term occurs in a chunk; IDF down-weights terms common across many "
          "chunks and up-weights rare, distinctive ones. Limitation: TF-IDF matches surface words, so two "
          "phrasings with the same meaning but different words score as unrelated.")

    # task 4 grounding evaluation, overlap 10 then overlap 0
    evaluate(overlap=10, label="overlap=10")
    print("  [ref 12/12, 12/12, 2/2]")
    evaluate(overlap=0, label="overlap=0")


if __name__ == "__main__":
    main()
