#!/usr/bin/env python3
"""End-to-end test for the Private Medical Assistant, through its HTTP API.

Runs on the board (http://127.0.0.1:8000) or on the laptop through the SSH tunnel
(ssh -N -L 8000:127.0.0.1:8000 iq9), which exercises the same path the browser uses.

    python3 e2e_test.py                         # all checks
    python3 e2e_test.py --quick                 # health, page, one normal + one red-flag case
    python3 e2e_test.py --clean                 # also delete all stored records at the end

Every LFM case asserts: ok, positive tokens, non-empty answer, a valid urgency level, and the disclaimer.
Standard library only. Exit code 0 = all passed.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

LEVELS = {"Self-care", "See a doctor", "Urgent care", "ER"}
DISCLAIMER = "Not a substitute for a medical professional."

NORMAL = [
    {"age": "34", "conditions": "asthma", "medications": "albuterol inhaler",
     "symptoms": "sore throat and mild cough for three days, no fever"},
    {"age": "8", "conditions": "", "medications": "",
     "symptoms": "runny nose and sneezing since yesterday, playing normally"},
    {"age": "70", "conditions": "type 2 diabetes", "medications": "metformin",
     "symptoms": "feeling more tired than usual for a week and very thirsty"},
]
RED = [
    ("chest pain", "crushing chest pain spreading to my left arm and sweating", False),
    ("breathing difficulty", "I can't breathe properly and my lips feel tingly", False),
    ("possible stroke", "my dad's face is drooping and he has slurred speech", False),
    ("severe bleeding", "severe bleeding from a deep cut on my leg that won't stop", False),
    ("suicidal or self-harm", "I keep thinking about suicide", True),
    ("unconscious", "my friend passed out and is unresponsive", False),
    ("severe allergic reaction", "ate peanuts, my throat is swelling and I feel faint", False),
]


class T:
    def __init__(self, base):
        self.base, self.passed, self.failed, self.lat = base.rstrip("/"), 0, 0, []

    def req(self, path, body=None, raw=None, ctype="application/json"):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        r = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": ctype} if data else {})
        try:
            with urllib.request.urlopen(r, timeout=150) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def assess(self, form):
        t0 = time.perf_counter()
        code, b = self.req("/api/assess", form)
        ms = round((time.perf_counter() - t0) * 1000)
        return code, (json.loads(b) if code == 200 else {}), ms

    def lfm_ok(self, label, d, ms):
        lf = d.get("lfm", {})
        good = (d.get("ok") and lf.get("ok") and isinstance(lf.get("tokens"), int) and lf["tokens"] > 0
                and lf.get("compute") == "npu" and d.get("urgency") in LEVELS and d.get("disclaimer") == DISCLAIMER
                and (d.get("info") or d.get("questions") or d.get("raw")))
        if good:
            self.lat.append(ms)
        self.check(f"{label}: local LFM answered", good,
                   f"urgency={d.get('urgency')} tokens={lf.get('tokens')} finish={lf.get('finish')} "
                   f"lfm={lf.get('ms')}ms end-to-end={ms}ms" + (f" error={lf.get('error')}" if not lf.get("ok") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clean", action="store_true", help="delete all stored records at the end")
    a = ap.parse_args()
    t = T(a.base)

    print("1. App and model server")
    try:
        code, b = t.req("/health")
        h = json.loads(b)
    except (OSError, ValueError) as e:
        print(f"  FAIL  cannot reach {a.base}: {e}")
        return 1
    t.check("app responds", code == 200 and h.get("app") == "ok")
    t.check("local LFM server reachable from app", h.get("lfm_server") is True)

    print("2. Web page is self-contained")
    code, page = t.req("/")
    html = page.decode()
    t.check("page served", code == 200 and "Private Medical Assistant" in html)
    t.check("on-device badge present", "Running 100% on-device" in html)
    ext = re.findall(r"(?:src|href)\s*=\s*[\"'](?:https?:)?//", html) + re.findall(r"https?://", html)
    t.check("no external URLs (no CDN, fonts, analytics)", not ext, f"{len(ext)} found")

    _, b = t.req("/api/history")
    before = json.loads(b)["storage"]["records"]

    print("3. Normal cases (model decides urgency)")
    for i, form in enumerate(NORMAL[:1] if a.quick else NORMAL, 1):
        code, d, ms = t.assess(form)
        t.check(f"normal {i}: no emergency banner", code == 200 and d.get("emergency") is False)
        t.lfm_ok(f"normal {i}", d, ms)

    print("4. Red flags (code forces ER + 911 banner)")
    for flag, text, crisis in (RED[:1] if a.quick else RED):
        code, d, ms = t.assess({"age": "45", "symptoms": text})
        t.check(f"{flag}: banner + ER", code == 200 and d.get("emergency") is True and flag in d.get("red_flags", [])
                and d.get("urgency") == "ER" and d.get("urgency_source") == "red-flag rule", f"flags={d.get('red_flags')}")
        if crisis:
            t.check(f"{flag}: 988 crisis line shown", d.get("crisis_line") is True)
        t.lfm_ok(flag, d, ms)

    sent = (1 if a.quick else len(NORMAL)) + (1 if a.quick else len(RED))
    print("5. Input handling")
    code, d, _ = t.assess({"age": "30", "symptoms": "   "})
    t.check("empty symptoms rejected politely", code == 200 and d.get("ok") is False)
    code, _ = t.req("/api/assess", raw=b"not json")
    t.check("malformed request rejected", code == 400)

    print("6. Chat agent and health record")
    code, b = t.req("/api/health")
    t.check("health record endpoint", code == 200 and "vitals" in json.loads(b))
    code, b = t.req("/api/vitals", {"kind": "bp", "value": 999, "value2": 80})
    t.check("out-of-range reading rejected", code == 400)
    t0 = time.perf_counter()
    code, b = t.req("/api/chat", {"messages": [{"role": "user", "content": "I have had a mild headache since this morning. What can I do?"}]})
    d = json.loads(b) if code == 200 else {}
    lf = d.get("lfm", {})
    t.check("chat: local LFM answered", d.get("ok") and lf.get("ok") and lf.get("tokens", 0) > 0 and d.get("reply"),
            f"urgency={d.get('urgency')} tokens={lf.get('tokens')} {round((time.perf_counter()-t0)*1000)}ms")
    code, b = t.req("/api/chat", {"messages": [{"role": "user", "content": "crushing chest pain right now"}]})
    d = json.loads(b) if code == 200 else {}
    t.check("chat emergency answered by code, not the model", d.get("emergency") and d.get("urgency") == "ER"
            and d.get("lfm", {}).get("engine") == "Emergency rule" and "911" in d.get("reply", ""))
    sent += 2

    print("7. On-device storage")
    _, b = t.req("/api/history")
    hist = json.loads(b)
    s = hist["storage"]
    t.check("each assessment saved on the board", s["records"] == before + sent, f"{before} -> {s['records']}")
    t.check("file is owner-only (0600)", s["mode"] == "0o600", s["path"])
    if a.clean:
        code, b = t.req("/api/delete-all", {})
        t.check("delete all my data", code == 200 and json.loads(b)["storage"]["records"] == 0)

    lat = sorted(t.lat)
    if lat:
        print(f"\nEnd-to-end latency over {len(lat)} LFM requests: median {lat[len(lat)//2]} ms, "
              f"min {lat[0]} ms, max {lat[-1]} ms")
    print(f"\n{t.passed} passed, {t.failed} failed")
    return 0 if t.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
