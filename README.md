# VoiceQL — Voice-to-SQL with Conversational Response Generation

Talk to a database. VoiceQL is a cascaded voice-to-SQL pipeline: you ask a question out loud, it transcribes your speech, generates and executes SQL against a relational database, then *speaks the answer back* in plain, conversational language.

This is a research prototype built as part of my MSc work on conversational database interfaces (Stellenbosch University, co-supervised with IBM Research Africa). It reimplements the cascaded architecture of Song & Wong (2022) and extends it with a novel spoken-response layer designed for multi-turn conversation.

> **[FILL IN: link to 2-minute demo video — record one!]**

## Why this is interesting

Most text-to-SQL research stops at generating correct SQL. But a *voice* interface has a harder problem: the answer must be **spoken**, which means raw result tables are useless. VoiceQL's response layer converts query results into a structured three-part spoken reply:

1. **Intent confirmation** — "You asked how many dogs are older than five."
2. **Plain-English query description** — no SQL jargon reaches the user's ears.
3. **Conversational insight** — patterns in the results, plus a suggested follow-up.

Combined with conversation memory, this supports multi-turn querying: follow-up questions like *"and how many of those are female?"* resolve against prior turns.

## Architecture

```
Mic ──> Silero VAD ──> Whisper STT ──> VoiceQLProcessor ──> ElevenLabs TTS ──> Speaker
                                            │
                          ┌─────────────────┴──────────────────┐
                          │ 1. SchemaLoader   (SQLite schema,   │
                          │                    cached at boot)  │
                          │ 2. PromptBuilder  (schema-aware,    │
                          │                    history-aware)   │
                          │ 3. SQLGenerator   (GPT-4o, retry    │
                          │                    loop on errors)  │
                          │ 4. QueryExecutor  (SQLite)          │
                          │ 5. ConversationMemory (multi-turn)  │
                          │ 6. ResponseGenerator (3-part spoken │
                          │                    response, GPT-4o)│
                          └─────────────────────────────────────┘
```

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) for real-time audio orchestration. SQL generation includes validation and an automatic retry loop (up to 3 attempts) that feeds execution errors back to the model for self-correction.

## Stack

- **Pipeline:** Pipecat (real-time audio frames, VAD-gated turn-taking)
- **STT:** OpenAI Whisper
- **VAD:** Silero
- **LLM:** GPT-4o (SQL generation + response generation), LangChain
- **TTS:** ElevenLabs (OpenAI TTS fallback)
- **Database:** SQLite — ships with the `pets_1` database from the [Spider](https://yale-lily.github.io/spider) text-to-SQL benchmark

## Quick start

```bash
pip install -r requirements.txt

# .env
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...        # optional, falls back to OpenAI TTS
SPIDER_DB_PATH=...            # optional, defaults to bundled pets_1

# Check your audio devices
python list_devices.py
python check_audio.py

# Run it
python voiceql.py
```

Then just speak: *"How many pets are in the database?"* … *"Which students own more than one?"*

## Example interaction

```
You:      How many dogs are there?
VoiceQL:  You wanted to know the number of dogs. I checked the pet records
          for dog entries. There are two dogs in the database — interesting
          that they're both among the older pets. Want me to break them
          down by age?
```

## Research context

The cascaded design (STT → text-to-SQL → execution → response → TTS) follows Song & Wong (2022). The contribution explored here is the **response-generation layer and conversation memory** — the parts that make the difference between a database that *executes* voice commands and one you can actually *converse* with. Evaluation work on multi-turn robustness ("conversation cliff" behaviour) is part of the ongoing MSc thesis and not included in this repository.

## Limitations

Prototype-grade: local audio only (no telephony/WebSocket transport), single-database sessions, no authentication, and SQL safety relies on validation + read-oriented prompting rather than a hardened sandbox. The thesis evaluation harness and datasets are not public.

## Author

**Takura Mukaro** — AI engineer working on voice and agentic systems.
[GitHub](https://github.com/taksrules) · [LinkedIn](https://linkedin.com/in/takura-mukaro-931a44210) · [takuramukaro.site](https://www.takuramukaro.site/)
