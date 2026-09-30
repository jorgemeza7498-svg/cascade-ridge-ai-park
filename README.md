# Cascade Ridge National Park - a walkable 3D world with a talking, learning AI bot

A browser-based 3D national park (procedural terrain, forests, a lake, day/night cycle) that you walk through with a bipedal character. A living habitat runs inside it: berry bushes feed rabbits, eagles hunt rabbits and fish, fish breed, and anglers fish the lake. **Bot**, an AI hiker, lives in the park, learns to move by copying you, evolves on its own, and talks with you using a local language model that remembers your conversations.

## What it demonstrates

- **Natural language processing:** every 12.9 seconds the game turns everything within 20 m of the bot (terrain, animals, water, you) into plain text and sends it to a local language model, which replies in first person. Replies are 50-350 words, configurable.
- **Long-term memory:** SQLite (built into Python) stores facts you tell the bot, shared events and a running summary, and recalls the relevant ones before each reply. Optional meaning-based recall with `sentence-transformers`.
- **CUDA / PyTorch:** the bot's movement network (23 inputs, 512 hidden neurons) trains on your first 30 seconds of play, then evolves. Fish, rabbit and eagle steering networks run as one batched GPU call.
- **Text-to-speech:** eSpeak NG reads the world report and the bot's reply aloud (button in the page).

## Tech stack

| Part | Tool |
|---|---|
| 3D world | Three.js (WebGL), procedural terrain, instanced meshes |
| Server | Python, Flask |
| Neural networks / CUDA | PyTorch |
| Language model | `openbmb/MiniCPM5-1B` (about 1B parameters) via Hugging Face Transformers, fp16 on GPU |
| Memory | SQLite (FTS5), optional sentence-transformers |
| Voice | eSpeak NG |

## Run it

1. Install Python 3.10+ and, for GPU use, a CUDA build of PyTorch (https://pytorch.org).
2. `pip install -r requirements.txt`
3. Install [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) (Windows: run the installer, then reopen your terminal).
4. Put `world_2.py` and `index_4.html` in one folder and run:

```
py world_2.py
```

5. Answer the terminal questions to tailor the bot (personality, what it calls you, topics, reply length, extra rules), then open http://127.0.0.1:5000. The first run downloads the language model (about 2 GB).

Options: `--defaults` skips the questions, `--customize` always asks.

## Controls

W A S D move, Shift run, Space jump, mouse look, V first/third person, B board the boat, T speed up time, M sound. Touch controls appear automatically on phones.

## In-game commands

Press **Enter twice** for the command line, **Enter three times** to talk to the bot.

| Command | What it does |
|---|---|
| `/findbot` | Beam of light from the sky on the bot for 12 s (blue by day, rose gold at night) |
| `/torch` | Flame in your hand for 30 s |
| `/resetbot` | Wipe the bot's brain and retrain on your next 30 s |
| `/followmeandconversation` | Bot loads `bot_movement.pth`, follows you (even into the boat) and chats; run again to stop |
| `/say <text>` | Talk to the bot |
| `/espeak` | Voice on/off (also key P) |
| `/remember <text>`, `/forget <text or all>`, `/memory` | Manage the bot's long-term memory |
| `/instructions [text, add text, clear]` | View or change custom instructions live |
| `/lmtest`, `/lmreload`, `/debug`, `/clearchat`, `/help` | Debugging and housekeeping |

The debug panel (top right) shows GPU status, whether the language model loaded, memory stats and more.

## Files created at runtime (not committed)

`bot_movement.pth` (bot brain), `bot_memory.db` (memory), `bot_conversations_log.json` (chat log), `bot_customization.json` (your terminal answers).

## Credits

Language model: OpenBMB MiniCPM. Libraries: Three.js, Flask, PyTorch, Hugging Face Transformers, SQLite, eSpeak NG. Built with the help of Claude (Anthropic).
