#!/usr/bin/env python3
"""Private Medical Assistant: an offline, on-device symptom helper for the IQ-9075.

Flow per request:
  1. Red-flag keyword check in code over the symptoms text (emergency banner, urgency forced to ER).
  2. One capped, validated call to the local LFM server (GenieX, 127.0.0.1:18181, NPU, enable_think false).
  3. Code parses the urgency level, enforces the four allowed levels, and guarantees the disclaimer.

Storage: each assessment (form + result) is appended to a JSONL file on the board only
(default ~/.local/share/private-medical-assistant/records.jsonl, dir 0700, file 0600). Not encrypted.
Nothing is sent anywhere except the loopback LFM server. --no-store disables saving; the UI can delete all.
The stdout log carries metadata only (no form content). Standard library only. Binds to loopback; open it from a laptop with SSH port forwarding.

    python3 app.py                    # serve http://127.0.0.1:8000
    python3 app.py --self-test        # red-flag and parser checks, no server needed
    python3 app.py --once "chest pain and sweating"   # one request through the LFM, prints JSON
"""

import argparse
import base64
import importlib.util
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "LiquidAI/LFM3-1.7B-GGUF:Q4_0"
LEVELS = ["Self-care", "See a doctor", "Urgent care", "ER"]
DISCLAIMER = "Not a substitute for a medical professional."
MAX_FIELD = 1500

# Red flags: deliberately broad. A false alarm (e.g. "no chest pain") is preferred to a miss.
RED_FLAGS = [
    ("chest pain", r"chest\s*(pain|pressure|tightness|hurts?)"),
    ("breathing difficulty", r"can'?t\s*breathe|can\s*not\s*breathe|cannot\s*breathe|unable to breathe|short(ness)?\s*of\s*breath"
                             r"|(difficulty|trouble|struggling)\s*breathing|not breathing"),
    ("possible stroke", r"face\s*(is\s*)?droop|facial\s*droop|droop(y|ing)\s*face|slurred|slurring"
                        r"|one[\s-]*sided\s*(weakness|numbness)|weak(ness)?\s*on\s*one\s*side|numb(ness)?\s*on\s*one\s*side"),
    ("severe bleeding", r"(severe|heavy|uncontrolled|profuse)\s*bleeding|bleeding\s*(heavily|a lot|won'?t stop|will not stop|that won'?t stop)"),
    ("suicidal or self-harm", r"suicid|self[\s-]*harm|kill(ing)?\s*myself|hurt(ing)?\s*myself|end(ing)?\s*my\s*life|want\s*to\s*die"),
    ("unconscious", r"unconscious|unresponsive|passed\s*out|not\s*waking\s*up|won'?t\s*wake"),
    ("severe allergic reaction", r"severe\s*allergic|anaphyla|throat\s*(is\s*)?(closing|swelling|swollen)"
                                 r"|swollen\s*(throat|tongue)|tongue\s*(is\s*)?swelling"),
]
_RED_RE = [(name, re.compile(p, re.I)) for name, p in RED_FLAGS]

SYSTEM_PROMPT = (
    "You are a careful health information assistant running offline on a private device. "
    "Give general information only. Never diagnose, never name a definite condition the person has, "
    "and never prescribe or change medication doses. Consider the person's age, conditions, and medications, "
    "and only link a symptom to a listed condition or medication when that link is well known. "
    "The ASK YOUR DOCTOR items are questions the person will ask their doctor, written in the first person "
    "(for example: 'Could my medication be causing this?'), not questions for the person. "
    "Reply in exactly this format and nothing else:\n"
    "URGENCY: <one of: Self-care | See a doctor | Urgent care | ER>\n"
    "INFO: <2-3 sentences of general information about what these symptoms can relate to>\n"
    "SELF-CARE:\n- <tip>\n- <tip>\n- <optional tip>\n"
    "ASK YOUR DOCTOR:\n- Could <first-person question I ask my doctor>?\n- Should I <first-person question>?\n"
    f"{DISCLAIMER}"
)


def strip_reasoning(text):
    """Drop <think> blocks and anything before a stray closing tag (seen on the board)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def normalize(text):
    return (text or "").replace("’", "'").replace("‘", "'").lower()


def red_flags(text):
    t = normalize(text)
    return [name for name, rx in _RED_RE if rx.search(t)]


def parse_answer(text):
    """Pull structured fields out of the model text. Missing pieces stay empty; code fills the rest."""
    text = strip_reasoning(text)
    out = {"urgency_model": None, "info": "", "self_care": [], "questions": []}
    m = re.search(r"URGENCY:\s*(.+)", text, re.I)
    if m:
        raw = m.group(1).strip().lower()
        for level in sorted(LEVELS, key=len, reverse=True):
            if level.lower() in raw:
                out["urgency_model"] = level
                break
        if out["urgency_model"] is None and ("emergency" in raw or raw.startswith("er")):
            out["urgency_model"] = "ER"
    m = re.search(r"INFO:\s*(.+?)(?=\n\s*(SELF-CARE|ASK YOUR DOCTOR)|\Z)", text, re.I | re.S)
    if m:
        out["info"] = m.group(1).replace(DISCLAIMER, "").strip()

    def bullets(header, stop):
        mm = re.search(header + r"\s*:?\s*\n(.+?)(?=" + stop + r"|\Z)", text, re.I | re.S)
        if not mm:
            return []
        items = [re.sub(r"^\s*([-*•]|\d+[.)])\s*", "", ln).strip() for ln in mm.group(1).splitlines()]
        return [i for i in items if i and DISCLAIMER.lower() not in i.lower()][:3]

    out["self_care"] = bullets(r"SELF-CARE", r"\n\s*ASK YOUR DOCTOR|\n\s*Not a substitute")
    out["questions"] = bullets(r"ASK YOUR DOCTOR", r"\n\s*Not a substitute")[:2]
    out["text"] = text
    return out


class LFM:
    """Capped, validated chat calls to the loopback GenieX server. Errors come back as data."""

    def __init__(self, server, max_tokens, timeout):
        self.server, self.max_tokens, self.timeout = server.rstrip("/"), max_tokens, timeout
        self.lock = threading.Lock()  # one generation at a time on the NPU

    def ready(self):
        try:
            with urllib.request.urlopen(self.server + "/v1/models", timeout=3) as r:
                return r.status == 200
        except OSError:
            return False

    def _call(self, messages):
        if not isinstance(self.max_tokens, int) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        body = {"model": MODEL, "messages": messages, "max_completion_tokens": self.max_tokens,
                "compute": "npu", "enable_think": False, "stream": False}
        req = urllib.request.Request(self.server + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.load(r)
        c = d["choices"][0]
        text = strip_reasoning((c.get("message") or {}).get("content") or "")
        return {"text": text, "tokens": d.get("usage", {}).get("completion_tokens", 0),
                "finish": c.get("finish_reason"), "ms": round((time.perf_counter() - t0) * 1000)}

    @staticmethod
    def _problem(r):
        if not isinstance(r["tokens"], int) or r["tokens"] <= 0:
            return "zero or missing completion tokens"
        if not r["text"]:
            return "empty answer"
        return None

    def ask(self, user):
        return self.ask_messages([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}])

    def ask_messages(self, messages):
        out = {"ok": False, "text": "", "tokens": 0, "finish": None, "ms": None, "retried": False, "error": None}
        messages = [dict(m) for m in messages]
        try:
            with self.lock:
                r = self._call(messages)
                err = self._problem(r)
                if err:  # retry once with a unique tag
                    out["retried"] = True
                    messages[-1]["content"] += f"\n[request:{uuid.uuid4().hex[:8]}]"
                    r = self._call(messages)
                    err = self._problem(r)
            out.update(r)
            out["ok"] = err is None
            out["error"] = err
        except urllib.error.HTTPError as e:
            out["error"] = f"HTTP {e.code}"
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError, IndexError) as e:
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out


DEFAULT_QUESTIONS = ["What could be causing these symptoms, and which warning signs should make me come back sooner?",
                     "Could any of my current conditions or medications be related to this?"]
_PATIENT_Q = re.compile(r"^(do|are|have|did|were|is there|how old|what is your|what are your|when did|how long have)\b", re.I)


def doctor_questions(items):
    """Keep first-person questions for the doctor; replace ones aimed at the patient (code, not the model)."""
    keep = [q for q in items if q.endswith("?") and not _PATIENT_Q.match(q) and " you " not in f" {q.lower()} "]
    for d in DEFAULT_QUESTIONS:
        if len(keep) >= 2:
            break
        keep.append(d)
    return keep[:2]


def clean(v, limit=MAX_FIELD):
    return str(v or "").strip()[:limit]


def assess(lfm, form):
    age_raw = clean(form.get("age"), 8)
    age = int(age_raw) if age_raw.isdigit() and 0 <= int(age_raw) <= 120 else None
    conditions, meds, symptoms = clean(form.get("conditions")), clean(form.get("medications")), clean(form.get("symptoms"))
    if not symptoms:
        return {"ok": False, "error": "Please describe your symptoms."}

    flags = red_flags(symptoms)
    user = (f"Age: {age if age is not None else 'not given'}\n"
            f"Existing conditions: {conditions or 'none given'}\n"
            f"Current medications: {meds or 'none given'}\n"
            f"Symptoms: {symptoms}")
    if flags:
        user += "\nNote: possible emergency signs were detected (" + ", ".join(flags) + "). Urgency must be ER."

    r = lfm.ask(user)
    parsed = parse_answer(r["text"]) if r["ok"] else {"urgency_model": None, "info": "", "self_care": [], "questions": [], "text": ""}

    # Code decides the final urgency: red flags force ER; otherwise the model's level if valid.
    if flags:
        urgency, source = "ER", "red-flag rule"
    elif parsed["urgency_model"]:
        urgency, source = parsed["urgency_model"], "model"
    else:
        urgency, source = "See a doctor", "default (model level missing)"

    return {
        "ok": True,
        "emergency": bool(flags),
        "red_flags": flags,
        "crisis_line": "suicidal or self-harm" in flags,
        "urgency": urgency,
        "urgency_source": source,
        "info": parsed["info"],
        "self_care": [] if flags else parsed["self_care"],
        "questions": doctor_questions(parsed["questions"]),
        "raw": parsed["text"] if r["ok"] and not (parsed["info"] or parsed["self_care"]) else "",
        "disclaimer": DISCLAIMER,
        "lfm": {"ok": r["ok"], "error": r["error"], "tokens": r["tokens"], "ms": r["ms"],
                "finish": r["finish"], "retried": r["retried"], "compute": "npu", "model": MODEL},
    }


CHAT_SYSTEM = (
    "You are Hushcare, a calm, friendly health information assistant running offline on a private device. "
    "You know the person's saved health record (below); use it to personalize answers, mention relevant readings or "
    "medications by name, and connect advice to their goals. Give general information and "
    "practical recommendations in plain language, at most about 120 words, in short paragraphs or up to 4 bullet "
    "points starting with '- '. Never diagnose, never say the person definitely has a condition, and never prescribe "
    "or change medication doses. If important details are missing, ask one short follow-up question. "
    "End every reply with one final line exactly in this form: URGENCY: <Self-care | See a doctor | Urgent care | ER>"
)
MAX_TURNS = 8


def profile_text(profile):
    if not isinstance(profile, dict):
        profile = {}
    age = clean(profile.get("age"), 8)
    age = age if age.isdigit() and int(age) <= 120 else ""
    parts = [f"age {age}" if age else "", clean(profile.get("conditions"), 300), clean(profile.get("medications"), 300)]
    labels = ["", "conditions: ", "medications: "]
    got = [lab + v for lab, v in zip(labels, parts) if v]
    return "; ".join(got) if got else "not given"


def split_urgency(text):
    """Remove the model's URGENCY line and any disclaimer it added; return (body, level or None)."""
    level = None
    lines = []
    for ln in strip_reasoning(text).splitlines():
        m = re.match(r"\s*\**\s*urgency\s*\**\s*:\s*(.+)", ln, re.I)
        if m:
            raw = m.group(1).strip().strip("*").lower()
            for lv in sorted(LEVELS, key=len, reverse=True):
                if lv.lower() in raw:
                    level = lv
                    break
            if level is None and ("emergency" in raw or raw.startswith("er")):
                level = "ER"
            continue
        if DISCLAIMER.lower().rstrip(".") in ln.lower():
            continue
        lines.append(ln.replace("**", ""))
    return "\n".join(lines).strip(), level


# ---- personal health record (profile + vitals), stored on the board only -------------------------------

VITALS = {  # kind: (label, unit, min, max)
    "bp": ("Blood pressure", "mmHg", 50, 260),
    "hr": ("Heart rate", "bpm", 25, 230),
    "glucose": ("Glucose", "mg/dL", 20, 600),
    "temp": ("Temperature", "\u00b0F", 90, 110),
    "weight": ("Weight", "lb", 40, 700),
    "sleep": ("Sleep", "h", 0, 24),
}
PROFILE_FIELDS = {"name": 60, "age": 3, "sex": 20, "height_in": 3, "weight_lb": 4, "conditions": 300,
                  "medications": 300, "allergies": 200, "goals": 300, "notes": 400}


def vital_status(kind, v, v2=None):
    """General reference ranges in code (not a diagnosis). Returns (level, label): ok | watch | alert."""
    if kind == "bp":
        if v >= 180 or v2 >= 120:
            return "alert", "Very high"
        if v < 90 or v2 < 60:
            return "watch", "Low"
        if v >= 140 or v2 >= 90:
            return "watch", "High"
        if v >= 130 or v2 >= 80:
            return "watch", "Elevated"
        return "ok", "Normal"
    if kind == "hr":
        return ("alert", "Very high") if v > 130 else ("alert", "Very low") if v < 40 else \
               ("watch", "High") if v > 100 else ("watch", "Low") if v < 50 else ("ok", "Normal")
    if kind == "glucose":
        return ("alert", "Very low") if v < 54 else ("alert", "Very high") if v > 250 else \
               ("watch", "Low") if v < 70 else ("watch", "High") if v > 140 else ("ok", "Normal")
    if kind == "temp":
        return ("alert", "High fever") if v >= 103 else ("alert", "Very low") if v < 95 else \
               ("watch", "Fever") if v >= 100.4 else ("ok", "Normal")
    if kind == "sleep":
        return ("watch", "Short") if v < 6 else ("watch", "Long") if v > 10 else ("ok", "Good")
    return "ok", ""


def bmi_of(profile):
    try:
        h, w = float(profile.get("height_in")), float(profile.get("weight_lb"))
        if not (36 <= h <= 96 and 40 <= w <= 700):
            return None
    except (TypeError, ValueError):
        return None
    b = round(703 * w / (h * h), 1)
    cat = "Underweight" if b < 18.5 else "Healthy" if b < 25 else "Overweight" if b < 30 else "Obesity range"
    return {"value": b, "category": cat, "level": "ok" if cat == "Healthy" else "watch"}


class Health:
    """profile.json + vitals.jsonl beside the assessment records; owner-only permissions; plaintext."""

    def __init__(self, store):
        self.store = store
        self.pfile = store.dir / "profile.json"
        self.vfile = store.dir / "vitals.jsonl"

    def _write(self, path, data, append=False):
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
        with self.store.lock:
            fd = os.open(path, flags, 0o600)
            try:
                os.write(fd, data.encode())
                os.fsync(fd)
            finally:
                os.close(fd)

    def profile(self):
        try:
            return json.loads(self.pfile.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_profile(self, body):
        if not isinstance(body, dict):
            raise ValueError("bad profile")
        prof = {k: clean(body.get(k), n) for k, n in PROFILE_FIELDS.items()}
        for k in ("age", "height_in", "weight_lb"):
            if prof[k] and not prof[k].replace(".", "", 1).isdigit():
                raise ValueError(f"{k} must be a number")
        prof["demo"] = bool(body.get("demo"))
        self._write(self.pfile, json.dumps(prof))
        return prof

    def add_vital(self, body):
        kind = body.get("kind")
        if kind not in VITALS:
            raise ValueError("unknown reading type")
        _, _, lo, hi = VITALS[kind]
        v = float(body.get("value"))
        if not lo <= v <= hi:
            raise ValueError(f"value out of range ({lo}-{hi})")
        rec = {"kind": kind, "value": v}
        if kind == "bp":
            v2 = float(body.get("value2"))
            if not 30 <= v2 <= 160:
                raise ValueError("diastolic out of range (30-160)")
            rec["value2"] = v2
        days = int(body.get("days_ago") or 0)
        if not 0 <= days <= 60:
            raise ValueError("days_ago out of range")
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - days * 86400))
        self._write(self.vfile, json.dumps(rec) + "\n", append=True)
        if kind == "weight":  # keep profile weight current
            prof = self.profile()
            if prof:
                prof["weight_lb"] = str(int(v) if v.is_integer() else v)
                self._write(self.pfile, json.dumps(prof))
        return rec

    def vitals(self):
        out = {k: [] for k in VITALS}
        try:
            for ln in self.vfile.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    out[r["kind"]].append(r)
                except (ValueError, KeyError):
                    continue
        except OSError:
            pass
        for k in out:
            out[k].sort(key=lambda r: r["ts"])
        return out

    def snapshot(self):
        prof, series = self.profile(), self.vitals()
        cards = []
        for k, (label, unit, _, _) in VITALS.items():
            pts = series[k][-7:]
            latest = pts[-1] if pts else None
            lvl, txt = vital_status(k, latest["value"], latest.get("value2")) if latest else (None, "")
            if k == "weight" and len(pts) >= 2:
                diff = round(pts[-1]["value"] - pts[0]["value"], 1)
                lvl, txt = "ok", ("Down %g lb" % -diff if diff < 0 else "Up %g lb" % diff if diff > 0 else "Steady")
            elif k == "weight" and latest:
                lvl, txt = "ok", "Tracked"
            cards.append({"kind": k, "label": label, "unit": unit, "latest": latest, "level": lvl, "status": txt,
                          "series": [[p["value"], p.get("value2")] for p in pts]})
        return {"profile": prof, "bmi": bmi_of(prof), "vitals": cards}

    def context(self):
        """Plain-text health record for the model's system message."""
        snap = self.snapshot()
        p = snap["profile"]
        if not p and not any(c["latest"] for c in snap["vitals"]):
            return "No health record saved."
        lines = []
        who = ", ".join(x for x in [p.get("name"), f"age {p['age']}" if p.get("age") else "", p.get("sex")] if x)
        if who:
            lines.append("Person: " + who)
        if snap["bmi"]:
            lines.append(f"BMI {snap['bmi']['value']} ({snap['bmi']['category']})")
        for k, lab in (("conditions", "Conditions"), ("medications", "Medications"), ("allergies", "Allergies"), ("goals", "Goals")):
            if p.get(k):
                lines.append(f"{lab}: {p[k]}")
        for c in snap["vitals"]:
            if c["latest"]:
                v = c["latest"]
                val = f"{v['value']:g}/{v['value2']:g}" if c["kind"] == "bp" else f"{v['value']:g}"
                trend = ""
                if len(c["series"]) >= 2:
                    first, last = c["series"][0][0], c["series"][-1][0]
                    trend = ", rising" if last > first * 1.03 else ", falling" if last < first * 0.97 else ", steady"
                lines.append(f"{c['label']}: {val} {c['unit']} on {v['ts'][:10]} ({c['status']}{trend}, {len(c['series'])} readings)")
        return "\n".join(lines)

    def delete(self):
        with self.store.lock:
            for f in (self.pfile, self.vfile):
                if f.exists():
                    f.unlink()


DEMO_PROFILE = {"name": "Alex Rivera (demo)", "age": "52", "sex": "Male", "height_in": "70", "weight_lb": "204",
                "conditions": "high blood pressure, prediabetes", "medications": "lisinopril 10 mg daily",
                "allergies": "penicillin", "goals": "lower my blood pressure, lose 15 lb, sleep better",
                "notes": "desk job, walks twice a week", "demo": True}
DEMO_VITALS = [("bp", 148, 94, 6), ("bp", 144, 92, 5), ("bp", 141, 90, 4), ("bp", 139, 88, 3), ("bp", 142, 91, 2),
               ("bp", 136, 87, 1), ("bp", 134, 86, 0), ("hr", 82, None, 6), ("hr", 78, None, 3), ("hr", 76, None, 0),
               ("glucose", 118, None, 5), ("glucose", 112, None, 2), ("glucose", 109, None, 0),
               ("weight", 207, None, 6), ("weight", 205.5, None, 3), ("weight", 204, None, 0),
               ("sleep", 5.5, None, 4), ("sleep", 6, None, 2), ("sleep", 5, None, 1), ("sleep", 6.5, None, 0),
               ("temp", 98.4, None, 0)]


class Vision:
    """Optional image route: ZETIC LFM3-VL-2B through the repo's run.py wrapper (fresh CLI per request)."""

    def __init__(self, run_py, sdk, model, tmp_dir, max_tokens=300, timeout=150):
        self.sdk, self.model = Path(sdk).expanduser(), Path(model).expanduser()
        self.tmp = Path(tmp_dir).expanduser()
        self.max_tokens, self.timeout = max_tokens, timeout
        self.mod, self.error = None, None
        try:
            spec = importlib.util.spec_from_file_location("zetic_run", Path(run_py).expanduser())
            self.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mod)
        except (OSError, ImportError, AttributeError, SyntaxError) as e:
            self.mod, self.error = None, f"{type(e).__name__}"

    def ready(self):
        return bool(self.mod) and (self.sdk / "bin/zetic_lfm3_cli").is_file() and (self.model / "v73/manifest.json").is_file()

    def ask(self, prompt, png):
        out = {"ok": False, "text": "", "tokens": 0, "finish": None, "ms": None, "retried": False, "error": None,
               "device": None, "load_ms": None}
        self.tmp.mkdir(parents=True, exist_ok=True)
        os.chmod(self.tmp, 0o700)
        path = self.tmp / f"img-{uuid.uuid4().hex}.png"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, png)
        finally:
            os.close(fd)
        t0 = time.perf_counter()
        try:
            r = self.mod.run(str(self.sdk), str(self.model), prompt, image=str(path),
                             max_tokens=self.max_tokens, timeout=self.timeout)
        except Exception as e:  # wrapper returns errors as data; this is a last guard
            r = {"ok": False, "errors": [f"{type(e).__name__}"]}
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        text = strip_reasoning(r.get("text") or "")
        tokens = r.get("completion_tokens") or 0
        out.update({"text": text, "tokens": tokens if isinstance(tokens, int) else 0, "finish": r.get("finish_reason"),
                    "device": r.get("device_evidence")})
        t = r.get("timings") or {}
        out["load_ms"] = t.get("load_ms") or t.get("model_load_ms")
        if r.get("ok") and out["tokens"] > 0 and text:
            out["ok"] = True
        else:
            errs = r.get("errors") or []
            out["error"] = (str(errs[0]) if errs else "empty or unusable answer")[:200]
        return out


def decode_png(data_url):
    """Accept only a 512x512 PNG data URL (the browser resizes); return bytes or raise ValueError."""
    if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
        raise ValueError("image must be a PNG data URL")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if len(raw) > 3_000_000 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError("not a valid PNG")
    w, h = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if (w, h) != (512, 512):
        raise ValueError("image must be 512x512")
    return raw


EMERGENCY_REPLY = (
    "This could be a medical emergency.\n"
    "- Call 911 now, or have someone take you to the nearest emergency department. Don't drive yourself.\n"
    "- If you're alone, unlock your door and stay on the line with the dispatcher.\n"
    "- Tell them your symptoms, when they started, your conditions, and your medications.\n"
    "- Don't wait to see if it passes."
)
CRISIS_REPLY = (
    "I'm really glad you told me. You deserve support right now.\n"
    "- Call or text 988 (Suicide & Crisis Lifeline) to talk to someone now, any time.\n"
    "- If you might act on these thoughts, call 911 or go to the nearest emergency department.\n"
    "- If you can, reach out to someone you trust and stay with them."
)
SAFE_MED_LINE = "- Don't start, stop, skip or change any medication without talking to your doctor or pharmacist first."
_MED_CHANGE = re.compile(
    r"\b(stop|stopping|quit|don'?t take|do not take|avoid taking|skip|hold off|double|increase|decrease|reduce|lower|cut|halve|change)\b"
    r"[^.\n]{0,50}\b(dose|dosage|medication|medicine|meds|pills?|tablets?|lisinopril|metformin|insulin|statin|warfarin|apixaban|inhaler)\b", re.I)


def guard_meds(reply):
    """Replace lines that tell the person to change medication with one safe line (code, not the model)."""
    out, replaced = [], False
    for ln in reply.splitlines():
        if _MED_CHANGE.search(ln):
            if not replaced:
                out.append(SAFE_MED_LINE)
                replaced = True
            continue
        out.append(ln)
    return "\n".join(out), replaced


def chat(lfm, vision, body, health=None):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return {"ok": False, "error": "No message."}
    convo = []
    for m in msgs[-MAX_TURNS:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return {"ok": False, "error": "Invalid message."}
        convo.append({"role": m["role"], "content": clean(m.get("content"))})
    if convo[-1]["role"] != "user":
        return {"ok": False, "error": "Last message must be from you."}
    user_text = convo[-1]["content"]
    image = body.get("image")
    if not user_text and not image:
        return {"ok": False, "error": "Type a question or attach a photo."}

    flags = red_flags(user_text)
    prof = health.context() if health else profile_text(body.get("profile"))
    note = ("\nNote: possible emergency signs were detected (" + ", ".join(flags) + "). URGENCY must be ER.") if flags else ""

    if flags:  # emergencies are answered by code only; the model is not consulted
        crisis = "suicidal or self-harm" in flags
        return {"ok": True, "reply": CRISIS_REPLY if crisis else EMERGENCY_REPLY, "urgency": "ER",
                "urgency_source": "red-flag rule", "emergency": True, "red_flags": flags, "crisis_line": crisis,
                "image": bool(image), "disclaimer": DISCLAIMER, "med_guard": False,
                "lfm": {"ok": True, "error": None, "tokens": 0, "ms": 0, "finish": None, "retried": False,
                        "engine": "Emergency rule", "compute": "code, no model"}}
    if image:
        if not vision or not vision.ready():
            return {"ok": False, "error": "The image model isn't installed on this board yet."}
        try:
            png = decode_png(image)
        except (ValueError, base64.binascii.Error) as e:
            return {"ok": False, "error": f"Image rejected: {e}"}
        prompt = (CHAT_SYSTEM + f"\nHealth record:\n{prof}" + note + "\nThe person attached a photo. "
                  + (f"Their question: {user_text}" if user_text else "Describe what is visible that matters for their health, and what they should do."))
        with lfm.lock:  # keep NPU models sequential
            r = vision.ask(prompt, png)
        route = {"engine": "ZETIC LFM3-VL-2B", "compute": "NPU (Hexagon v73, 2 NSP)"}
    else:
        messages = [{"role": "system", "content": CHAT_SYSTEM + f"\nHealth record:\n{prof}"}] + convo
        if note:
            messages[-1] = {"role": "user", "content": user_text + note}
        r = lfm.ask_messages(messages)
        route = {"engine": "GenieX LFM3-1.7B Q4_0", "compute": "NPU (HTP0)"}

    reply, level = split_urgency(r["text"]) if r["ok"] else ("", None)
    reply, med_guard = guard_meds(reply)
    if flags:
        urgency, source = "ER", "red-flag rule"
    elif level:
        urgency, source = level, "model"
    else:
        urgency, source = None, None
    return {"ok": True, "reply": reply, "urgency": urgency, "urgency_source": source, "emergency": bool(flags),
            "red_flags": flags, "crisis_line": "suicidal or self-harm" in flags, "image": bool(image),
            "disclaimer": DISCLAIMER, "med_guard": med_guard,
            "lfm": {"ok": r["ok"], "error": r["error"], "tokens": r["tokens"], "ms": r["ms"], "finish": r["finish"],
                    "retried": r["retried"], **route}}


class Store:
    """Append-only local JSONL with owner-only permissions. Plaintext: protected by the board account only."""

    def __init__(self, data_dir, enabled=True):
        self.enabled = enabled
        self.dir = Path(data_dir).expanduser()
        self.path = self.dir / "records.jsonl"
        self.lock = threading.Lock()
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.dir, 0o700)

    def add(self, form, res):
        if not self.enabled:
            return None
        rec = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "input": {k: clean(form.get(k), 8 if k == "age" else MAX_FIELD) for k in ("age", "conditions", "medications", "symptoms")},
               "result": {k: res[k] for k in ("urgency", "urgency_source", "red_flags", "info", "self_care", "questions")},
               "lfm": res["lfm"]}
        line = (json.dumps(rec) + "\n").encode()
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        return rec["id"]

    def add_chat(self, body, res):
        if not self.enabled:
            return None
        msgs = body.get("messages") or [{}]
        rec = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "chat",
               "session": clean(body.get("session"), 40),
               "input": {"symptoms": clean(msgs[-1].get("content")) or "(photo)", "image_attached": res["image"],
                         "profile": profile_text(body.get("profile"))},
               "result": {"urgency": res["urgency"] or "Chat", "urgency_source": res["urgency_source"],
                          "red_flags": res["red_flags"], "info": res["reply"], "self_care": [], "questions": []},
               "lfm": res["lfm"]}
        line = (json.dumps(rec) + "\n").encode()
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        return rec["id"]

    def history(self, limit=20):
        if not self.enabled or not self.path.exists():
            return []
        with self.lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        out = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def info(self):
        exists = self.enabled and self.path.exists()
        count = sum(1 for _ in self.path.open(encoding="utf-8")) if exists else 0
        return {"enabled": self.enabled, "path": str(self.path), "records": count,
                "bytes": self.path.stat().st_size if exists else 0,
                "mode": oct(self.path.stat().st_mode & 0o777) if exists else None, "encrypted": False}

    def delete_all(self):
        with self.lock:
            if self.path.exists():
                self.path.unlink()
        return self.info()


def log(event):
    # Metadata only; never form contents.
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}), flush=True)


def make_handler(lfm, page, store, vision=None, classic=b"", health=None):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' blob: data:")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html")
            elif self.path == "/classic":
                self._send(200, classic, "text/html")
            elif self.path == "/health":
                self._send(200, {"app": "ok", "lfm_server": lfm.ready(), "vision": bool(vision and vision.ready())})
            elif self.path == "/api/health":
                self._send(200, health.snapshot())
            elif self.path == "/api/history":
                self._send(200, {"storage": store.info(), "records": store.history()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path in ("/api/profile", "/api/vitals", "/api/demo-profile"):
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self._send(400, {"ok": False, "error": "Invalid request."})
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n)) if 0 < n <= 20000 else {}
                    if self.path == "/api/profile":
                        health.save_profile(body)
                    elif self.path == "/api/vitals":
                        health.add_vital(body)
                    else:
                        health.delete()
                        health.save_profile(DEMO_PROFILE)
                        for kind, v, v2, d in DEMO_VITALS:
                            health.add_vital({"kind": kind, "value": v, "value2": v2, "days_ago": d})
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    return self._send(400, {"ok": False, "error": str(e)[:120] or "Invalid data."})
                except OSError as e:
                    return self._send(500, {"ok": False, "error": f"could not save ({type(e).__name__})"})
                log({"event": self.path.rsplit("/", 1)[1]})
                return self._send(200, {"ok": True, **health.snapshot()})
            if self.path == "/api/delete-all":
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self._send(400, {"ok": False, "error": "Invalid request."})
                health.delete()
                info = store.delete_all()
                log({"event": "delete_all", "records_after": info["records"]})
                return self._send(200, {"ok": True, "storage": info})
            if self.path not in ("/api/assess", "/api/chat"):
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > (4_500_000 if self.path == "/api/chat" else 20000):
                    raise ValueError("bad length")
                form = json.loads(self.rfile.read(n))
                if not isinstance(form, dict):
                    raise ValueError("bad body")
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"ok": False, "error": "Invalid request."})
            t0 = time.perf_counter()
            if self.path == "/api/chat":
                try:
                    res = chat(lfm, vision, form, health)
                except Exception as e:
                    res = {"ok": False, "error": f"internal error: {type(e).__name__}"}
                if res.get("ok"):
                    try:
                        res["record_id"] = store.add_chat(form, res)
                    except OSError as e:
                        res["record_id"], res["store_error"] = None, type(e).__name__
                    log({"event": "chat", "image": res["image"], "stored": bool(res.get("record_id")),
                         "red_flags": res["red_flags"], "urgency": res["urgency"], "engine": res["lfm"]["engine"],
                         "lfm_ok": res["lfm"]["ok"], "lfm_error": res["lfm"]["error"], "tokens": res["lfm"]["tokens"],
                         "lfm_ms": res["lfm"]["ms"], "total_ms": round((time.perf_counter() - t0) * 1000)})
                else:
                    log({"event": "chat_error", "error": res.get("error")})
                return self._send(200, res)
            try:
                res = assess(lfm, form)
            except Exception as e:  # keep serving; report as data
                res = {"ok": False, "error": f"internal error: {type(e).__name__}"}
            if res.get("ok"):
                try:
                    res["record_id"] = store.add(form, res)
                except OSError as e:
                    res["record_id"], res["store_error"] = None, f"{type(e).__name__}"
                log({"event": "assess", "stored": bool(res.get("record_id")), "red_flags": res["red_flags"], "urgency": res["urgency"],
                     "urgency_source": res["urgency_source"], "lfm_ok": res["lfm"]["ok"], "lfm_error": res["lfm"]["error"],
                     "tokens": res["lfm"]["tokens"], "lfm_ms": res["lfm"]["ms"], "finish": res["lfm"]["finish"],
                     "total_ms": round((time.perf_counter() - t0) * 1000)})
            else:
                log({"event": "assess_error", "error": res.get("error")})
            self._send(200, res)

    return H


def self_test():
    cases = {
        "I have chest pain and sweating": ["chest pain"],
        "I can't breathe well": ["breathing difficulty"],
        "shortness of breath when walking": ["breathing difficulty"],
        "my dad's face is drooping and he has slurred speech": ["possible stroke"],
        "severe bleeding from a cut": ["severe bleeding"],
        "I have been thinking about suicide": ["suicidal or self-harm"],
        "she passed out and is unresponsive": ["unconscious"],
        "severe allergic reaction, throat swelling": ["severe allergic reaction"],
        "mild headache and runny nose for two days": [],
        "sore throat and low fever": [],
    }
    fails = 0
    for text, want in cases.items():
        got = red_flags(text)
        ok = all(w in got for w in want) and (want or not got)
        fails += not ok
        print(("PASS" if ok else "FAIL"), repr(text), "->", got)
    sample = ("URGENCY: See a doctor\nINFO: Headaches can relate to many causes.\nSELF-CARE:\n- Rest\n- Drink water\n"
              "ASK YOUR DOCTOR:\n- Could my medication cause this?\n- When should I worry?\n" + DISCLAIMER)
    p = parse_answer(sample)
    ok = p["urgency_model"] == "See a doctor" and len(p["self_care"]) == 2 and len(p["questions"]) == 2 and p["info"]
    fails += not ok
    print(("PASS" if ok else "FAIL"), "parse_answer", {k: p[k] for k in ("urgency_model", "self_care", "questions")})
    body, lv = split_urgency("Rest and fluids help.\n- Drink water\nURGENCY: See a doctor\nNot a substitute for a medical professional.")
    ok = lv == "See a doctor" and "URGENCY" not in body and "substitute" not in body and "- Drink water" in body
    fails += not ok
    print(("PASS" if ok else "FAIL"), "split_urgency", lv, repr(body))
    ok = (vital_status("bp", 185, 100)[0] == "alert" and vital_status("bp", 142, 91)[1] == "High"
          and vital_status("bp", 118, 76)[1] == "Normal" and vital_status("glucose", 50)[0] == "alert"
          and bmi_of({"height_in": "70", "weight_lb": "204"})["value"] == 29.3)
    fails += not ok
    print(("PASS" if ok else "FAIL"), "vital_status/bmi")
    g, hit = guard_meds("- Rest well.\n- Do not take lisinopril again until stable.\n- Stop your metformin dose.")
    ok = hit and "lisinopril" not in g and "metformin" not in g and g.count("pharmacist") == 1 and "Rest well" in g
    g2, hit2 = guard_meds("- Keep taking your lisinopril as prescribed.\n- Reduce sodium to under 1,500 mg a day.")
    ok = ok and not hit2 and "lisinopril" in g2
    fails += not ok
    print(("PASS" if ok else "FAIL"), "guard_meds")
    q = doctor_questions(["How old are you?", "Could my inhaler be making this worse?"])
    ok = q[0] == "Could my inhaler be making this worse?" and q[1] == DEFAULT_QUESTIONS[0]
    fails += not ok
    print(("PASS" if ok else "FAIL"), "doctor_questions", q)
    print("all passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lfm", default="http://127.0.0.1:18181")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--data-dir", default="~/.local/share/private-medical-assistant")
    ap.add_argument("--no-store", action="store_true", help="do not save assessments")
    ap.add_argument("--vision-run", default="~/zetic-demo/run.py", help="ZETIC run.py wrapper (image route)")
    ap.add_argument("--vision-sdk", default="~/zetic-demo/sdk")
    ap.add_argument("--vision-model", default="~/zetic-demo/models/lfm3-vl-2b")
    ap.add_argument("--no-vision", action="store_true", help="turn off the photo route")
    ap.add_argument("--self-test", action="store_true", help="red-flag/parser checks; no server")
    ap.add_argument("--once", metavar="SYMPTOMS", help="one request through the LFM, print JSON, exit")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.max_tokens <= 0:
        ap.error("--max-tokens must be positive")
    lfm = LFM(a.lfm, a.max_tokens, a.timeout)
    if a.once:
        res = assess(lfm, {"symptoms": a.once})
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("ok") and res["lfm"]["ok"] else 2)
    page = (HERE / "index.html").read_bytes()
    store = Store(a.data_dir, enabled=not a.no_store)
    vision = None if a.no_vision else Vision(a.vision_run, a.vision_sdk, a.vision_model, Path(a.data_dir).expanduser() / "tmp")
    classic = (HERE / "classic.html").read_bytes() if (HERE / "classic.html").exists() else b"classic page missing"
    health = Health(store)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(lfm, page, store, vision, classic, health))
    log({"event": "start", "url": f"http://{a.host}:{a.port}", "lfm_server": a.lfm, "lfm_ready": lfm.ready(),
         "max_tokens": a.max_tokens, "storage": store.info(), "vision_ready": bool(vision and vision.ready())})
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
