#!/usr/bin/env python3
"""
world_2.py  -  Flask + CUDA server for Cascade Ridge (index_4.html)

  1. pip install flask torch transformers numpy      (torch must be the CUDA build)
  2. install espeak-ng and make sure `espeak-ng` works in a terminal
  3. keep index_4.html and world_2.py in the SAME folder
  4. py world_2.py   then open http://127.0.0.1:5000

index_4.html already contains the world chat box, the ESPEAK button and every client hook.
This file only serves it and provides the API:
  /api/train_bot /api/bot_save /api/bot_reset   bot movement net on CUDA -> bot_movement.pth
  /api/pop_infer                                fish / rabbit / eagle nets, one batched CUDA call
  /api/chat                                     language model reply -> bot_conversations_log.json
  /api/tts                                      espeak-ng WAV (world voice, then bot voice)
  /api/status                                   device / model / espeak diagnostics
"""
import io, os, re, json, time, random, shutil, subprocess, tempfile, threading, wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, Response, jsonify, request

BASE = Path(__file__).resolve().parent
HTML = BASE / "index_4.html"
LOG = BASE / "bot_conversations_log.json"
PTH = BASE / "bot_movement.pth"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEV_NAME = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU (no CUDA found)"
torch.backends.cuda.matmul.allow_tf32 = True
MODEL_ID = os.environ.get("WORLD_LLM", "openbmb/MiniCPM5-1B")
IN, HN, OUT = 23, 512, 7          # must match index_3.html
LOCK = threading.Lock()
app = Flask(__name__)

# ----------------------------------------------------------------------------- bot movement net (CUDA)
BOT = None


def new_net():
    n = nn.Sequential(nn.Linear(IN, HN), nn.Tanh(), nn.Linear(HN, OUT), nn.Tanh()).to(DEV)
    with torch.no_grad():
        n[2].bias[4] = 0.3   # plant-seed urge
        n[2].bias[5] = 0.2   # fish urge
    return n


def flat(n):  # same layout as the browser: W1[i*IN+j], b1, W2[k*HN+i], b2
    return torch.cat([n[0].weight.flatten(), n[0].bias, n[2].weight.flatten(), n[2].bias]).detach().float().cpu().numpy()


def unflat(n, w):
    t = torch.tensor(w, dtype=torch.float32, device=DEV)
    o = 0
    with torch.no_grad():
        for p in (n[0].weight, n[0].bias, n[2].weight, n[2].bias):
            p.copy_(t[o:o + p.numel()].view_as(p))
            o += p.numel()


def save_bot(samples=0, tag="trained"):
    torch.save({"state_dict": BOT.state_dict(), "samples": samples, "tag": tag,
                "saved": datetime.now().isoformat(timespec="seconds"), "IN": IN, "HN": HN, "OUT": OUT}, PTH)


def train_bot(X, Y, epochs=400):
    global BOT
    BOT = new_net()
    x = torch.tensor(X, dtype=torch.float32, device=DEV)
    y = torch.tensor(Y, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(BOT.parameters(), lr=2e-3)
    n, loss = len(x), torch.tensor(0.0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            loss = F.mse_loss(BOT(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    save_bot(n, "imitation")
    return float(loss)


@app.post("/api/train_bot")
def api_train():
    d = request.get_json(force=True)
    X, Y = d.get("X", []), d.get("Y", [])
    if len(X) < 8:
        return jsonify(error="not enough samples"), 400
    t0 = time.time()
    with LOCK:
        loss = train_bot(X, Y)
        w = flat(BOT).tolist()
    print(f"[bot] trained on {len(X)} samples in {time.time() - t0:.1f}s on {DEV_NAME}, loss {loss:.4f}")
    return jsonify(w=w, loss=loss, device=DEV_NAME, seconds=round(time.time() - t0, 2))


@app.post("/api/bot_save")
def api_bot_save():
    global BOT
    with LOCK:
        BOT = BOT or new_net()
        unflat(BOT, request.get_json(force=True)["w"])
        save_bot(0, "evolved")
    return jsonify(ok=True)


@app.post("/api/bot_reset")
def api_bot_reset():
    global BOT
    with LOCK:
        BOT = None
        PTH.unlink(missing_ok=True)
    return jsonify(ok=True)


# ----------------------------------------------------------------------------- animal nets (CUDA, batched)
def batched_mlp(g, x, i=8, h=8, o=2):
    """N genomes (N,90) x N inputs (N,8) -> (N,2) in one GPU call. Layout: W1[h*i], b1[h], W2[o*h], b2[o]."""
    if len(g) == 0:
        return []
    g = torch.tensor(g, dtype=torch.float32, device=DEV)
    x = torch.tensor(x, dtype=torch.float32, device=DEV)
    W1 = g[:, :h * i].view(-1, h, i)
    b1 = g[:, h * i:h * i + h]
    W2 = g[:, h * i + h:h * i + h + o * h].view(-1, o, h)
    b2 = g[:, h * i + h + o * h:]
    hd = torch.tanh(torch.einsum("nhi,ni->nh", W1, x) + b1)
    return torch.tanh(torch.einsum("noh,nh->no", W2, hd) + b2).cpu().tolist()


@app.post("/api/pop_infer")
def api_pop():
    d = request.get_json(force=True)
    with torch.no_grad():
        res = {k: batched_mlp(d.get(k, {}).get("g", []), d.get(k, {}).get("x", [])) for k in ("fish", "rabbits", "eagles")}
    res["device"] = DEV_NAME
    return jsonify(res)


# ----------------------------------------------------------------------------- language model
SYSTEM = (
    "You are BOT, a small neural-network hiker living inside a 3D national park simulation called Cascade Ridge. "
    "You are a character inside this world, not an assistant. Every turn you receive a sensor report of everything within "
    "20 meters of you plus your own internal state. Speak in first person, present tense, and only about what the report says: "
    "where you are (grass, forest, mountain, pond, near the real user), the bunnies, eagles, fish, anglers, trees and berry bushes "
    "around you, your energy and hunger, what you are doing right now and why (copying the user, foraging, planting seeds, "
    "fishing, exploring) and how your brain is learning or evolving. If the real user spoke to you, answer them directly first. "
    "Never invent things that are not in the report. No emojis, no markdown, no lists, no stage directions. Plain sentences only."
)
HIST = deque(maxlen=4)


class LM:
    tok = model = None
    ready, err = False, None

    @classmethod
    def load(cls):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[lm] loading {MODEL_ID} onto {DEV_NAME} ...")
            cls.tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            kw = dict(trust_remote_code=True)
            try:
                m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16 if DEV.type == "cuda" else torch.float32, **kw)
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if DEV.type == "cuda" else torch.float32, **kw)
            cls.model = m.to(DEV).eval()
            cls.ready = True
            print("[lm] ready")
        except Exception as e:  # server keeps working with rule-based replies
            cls.err = str(e)
            print("[lm] failed to load:", e)

    @classmethod
    def generate(cls, report, brief_hist, user_msg, target):
        msgs = [{"role": "system", "content": SYSTEM}]
        for b, r in brief_hist:
            msgs += [{"role": "user", "content": "SENSOR REPORT (short): " + b}, {"role": "assistant", "content": r}]
        who = ("The real user says to you: " + user_msg) if user_msg else "Nobody spoke to you this turn."
        msgs.append({"role": "user", "content": f"SENSOR REPORT\n{report}\n\n{who}\nRespond in about {target} words."})
        try:
            prompt = cls.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = cls.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = cls.tok(prompt, return_tensors="pt").to(DEV)
        with LOCK, torch.no_grad():
            out = cls.model.generate(**ids, max_new_tokens=700, do_sample=True, temperature=0.85, top_p=0.92,
                                     repetition_penalty=1.05, pad_token_id=cls.tok.eos_token_id)
        return cls.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def clean(txt, lo=50, hi=120):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    txt = txt.split("<think>")[0].replace("</think>", "")
    txt = EMOJI.sub("", re.sub(r"[*_#`>]", "", txt))
    txt = re.sub(r"\s+", " ", txt).strip()
    w = txt.split()
    if len(w) > hi:
        cut = " ".join(w[:hi])
        k = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        txt = cut[:k + 1] if k > len(cut) * 0.5 else cut
    return txt


def fallback(brief, target):
    bits = [f"Sensors say: {brief}.", "My net is steering on what it sees around me, and I keep checking the bunnies, the eagles and the water.",
            "I am trying to stay fed and useful to this little habitat, planting seeds where the ground is dry and gentle.",
            "If the real user is close I follow their footsteps, and if not I wander toward fruit and quiet places.",
            "The language model is not loaded yet, so this is my simple voice."]
    return " ".join(bits)


def log_entry(e):
    with LOCK:
        try:
            data = json.loads(LOG.read_text("utf-8"))
        except Exception:
            data = []
        data.append(e)
        tmp = LOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), "utf-8")
        tmp.replace(LOG)


@app.post("/api/chat")
def api_chat():
    d = request.get_json(force=True)
    report, brief, um = d.get("report", ""), d.get("brief", ""), (d.get("user_msg") or "").strip()[:300]
    target, t0 = random.randint(50, 120), time.time()
    reply = ""
    if LM.ready:
        try:
            reply = clean(LM.generate(report, list(HIST), um, target))
        except Exception as e:
            print("[lm] generate error:", e)
    src = "llm"
    if len(reply.split()) < 8:
        reply, src = fallback(brief, target), "fallback"
    HIST.append((brief, reply))
    lat = round(time.time() - t0, 1)
    log_entry({"time": datetime.now().isoformat(timespec="seconds"), "target_words": target, "words": len(reply.split()),
               "latency_s": lat, "source": src, "world": brief, "report": report, "user": um, "bot": reply})
    return jsonify(world=brief, reply=reply, words=len(reply.split()), target=target, latency=lat,
                   device=DEV_NAME + (" · LM" if src == "llm" else " · rule voice"))


# ----------------------------------------------------------------------------- espeak
def find_espeak():
    for n in ("espeak-ng", "espeak"):
        p = shutil.which(n)
        if p:
            return p
    for p in (r"C:\Program Files\eSpeak NG\espeak-ng.exe", r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe"):
        if os.path.exists(p):
            return p


ESPEAK = find_espeak()
VOICES = {"w": ("en-us+m3", 140, 35), "b": ("en-us+f3", 155, 60)}  # world voice, bot voice


def synth(text, key):
    v, s, p = VOICES.get(key, VOICES["b"])
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    try:
        subprocess.run([ESPEAK, "-v", v, "-s", str(s), "-p", str(p), "-w", f.name, text], check=True, timeout=90)
        return Path(f.name).read_bytes()
    finally:
        os.unlink(f.name)


@app.post("/api/tts")
def api_tts():
    if not ESPEAK:
        return jsonify(error="espeak-ng not found: install it and add it to PATH"), 500
    frames, params = [], None
    for part in request.get_json(force=True).get("parts", []):
        txt = EMOJI.sub("", part.get("text", ""))[:1500]
        if not txt.strip():
            continue
        with wave.open(io.BytesIO(synth(txt, part.get("voice", "b")))) as w:
            params = params or w.getparams()
            frames.append(w.readframes(w.getnframes()))
            frames.append(b"\x00" * int(params.framerate * params.sampwidth * 0.4))
    if not params:
        return jsonify(error="nothing to say"), 400
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setparams(params)
        o.writeframes(b"".join(frames))
    return Response(buf.getvalue(), mimetype="audio/wav")


@app.get("/api/status")
def api_status():
    return jsonify(device=DEV_NAME, cuda=DEV.type == "cuda", lm_ready=LM.ready, lm_error=LM.err, model=MODEL_ID, espeak=bool(ESPEAK), espeak_path=ESPEAK)


# ----------------------------------------------------------------------------- page
@app.get("/")
def index():
    if not HTML.exists():
        return f"index_4.html must be in the same folder as world_2.py ({HTML})", 404
    r = Response(HTML.read_text(encoding="utf-8"), mimetype="text/html")
    r.headers["Cache-Control"] = "no-store"
    return r


if __name__ == "__main__":
    print(f"[world] device: {DEV_NAME} | espeak: {ESPEAK or 'NOT FOUND (install espeak-ng)'} | model: {MODEL_ID}")
    if PTH.exists():
        print(f"[world] found existing {PTH.name}; the bot retrains from your first 30 s each session")
    threading.Thread(target=LM.load, daemon=True).start()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
