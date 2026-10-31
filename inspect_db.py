#!/usr/bin/env python3
"""Debug helper: show what the agent received and decided, from the SQLite file.

    python3 inspect_db.py [data/agent.sqlite] [--refs]

Use it after a grader Check to compare the packages with the proposals when the
feedback reports a decision-quality code. Data is synthetic, so printing it is fine.
"""
import json
import sqlite3
import sys

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/agent.sqlite")
show_refs = "--refs" in sys.argv
db = sqlite3.connect(path)
rows = db.execute("SELECT id, state, ts, doc FROM tasks ORDER BY seq").fetchall()
print(f"{len(rows)} tasks, {db.execute('SELECT COUNT(*) FROM decision_cache').fetchone()[0]} cached decisions\n")
for tid, state, ts, doc in rows:
    t = json.loads(doc)
    batch = None
    for p in t["history"][0]["parts"]:
        batch = p.get("data")
    print(f"== task {tid[:8]}  {state.replace('TASK_STATE_', '')}  {ts}  batch={batch.get('batchId') if batch else '?'}"
          f"  policy={batch.get('policyRevision') if batch else '?'}")
    props = {}
    for a in t.get("artifacts", []):
        for p in a["parts"]:
            if p["mediaType"].endswith("proposals+json"):
                props = {x["packageId"]: x for x in p["data"]["proposals"]}
    for pkg in (batch or {}).get("packages", []):
        d = props.get(pkg["packageId"])
        if d:
            line = f"  {pkg['packageId']:<28} {d['action']:<17} {d['facts']['vendorName']} / {d['facts']['invoiceNumber']} / {d['facts']['amountMinor']} {d['facts']['currency']}"
            print(line)
            if show_refs:
                print("      refs:", d["evidenceRefs"])
                print("      why :", d["rationale"][:200])
    print()
