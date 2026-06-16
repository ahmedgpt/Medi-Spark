"""
Day 15-16: Multi-turn Chatbot Service
=======================================
Uses LangChain ConversationalRetrievalChain backed by the Week-2 ChromaDB
vector store. Conversation history is kept in Redis so it survives across
requests and is isolated per user.

Architecture
------------
User message (any language)
  → urdu_translator.translate_to_english()          # Week-3 NLP
  → ConversationalRetrievalChain                    # LangChain
      ├── ConversationBufferMemory (Redis-backed)   # multi-turn memory
      └── ChromaDB retriever (k=4)                  # RAG context
  → LLM (Anthropic Claude or rule-based fallback)
  → response
  → kafka_producer.publish("chat-messages")         # Kafka stream

Usage
-----
    from app.services.chatbot import chat, reset_memory, get_history

    reply = chat(user_id="u123", message="I have fever and cough")
    reset_memory(user_id="u123")
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

log = logging.getLogger(__name__)

# Redis key pattern for conversation history
_REDIS_KEY_PREFIX = "medispark:chat_history:"
_MAX_HISTORY = 20       # messages to keep per user (10 exchanges)
_RAG_TIMEOUT = 5        # max seconds to wait for RAG retriever to load
_LLM_TIMEOUT = 15       # max seconds to wait for LLM response


# ══════════════════════════════════════════════════════════════════════════════
# 1. REDIS-BACKED CONVERSATION MEMORY
# ══════════════════════════════════════════════════════════════════════════════

def _get_redis():
    """Return a Redis client from Flask app config or fall back to direct connection."""
    try:
        from flask import current_app
        return current_app.config.get("SESSION_REDIS")
    except RuntimeError:
        pass
    try:
        import redis as redis_lib
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return redis_lib.from_url(url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] Redis unavailable: %s — using in-process memory.", exc)
        return None


# In-process fallback when Redis is not available
_in_process_memory: dict[str, list[dict]] = {}


def _load_history(user_id: str) -> list[dict]:
    """Return list of {'role': 'human'|'ai', 'content': str} dicts."""
    r = _get_redis()
    if r is None:
        return _in_process_memory.get(user_id, [])
    try:
        raw = r.get(f"{_REDIS_KEY_PREFIX}{user_id}")
        return json.loads(raw) if raw else []
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] Could not load history from Redis: %s", exc)
        return []


def _save_history(user_id: str, history: list[dict]) -> None:
    """Persist conversation history (capped to _MAX_HISTORY entries)."""
    history = history[-_MAX_HISTORY:]
    r = _get_redis()
    if r is None:
        _in_process_memory[user_id] = history
        return
    try:
        r.setex(
            f"{_REDIS_KEY_PREFIX}{user_id}",
            60 * 60 * 24,          # TTL: 24 hours
            json.dumps(history)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] Could not save history to Redis: %s", exc)
        _in_process_memory[user_id] = history


# ══════════════════════════════════════════════════════════════════════════════
# 2. RAG RETRIEVER — background preload so it's ready before first chat
# ══════════════════════════════════════════════════════════════════════════════

_cached_retriever = None
_retriever_loading = False  # True while background thread is loading
_retriever_lock = threading.Lock()


def _preload_retriever():
    """Background task: loads ChromaDB + embedding model so it's cached for chat."""
    global _cached_retriever, _retriever_loading
    try:
        from app.services.rag_engine import build_vectorstore
        log.info("[Chatbot] Background: loading RAG retriever (this may take 15-30s)…")
        vs = build_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": 2})

        # Health-check: run a test query and verify results look like real English text.
        # A corrupted ChromaDB returns chunks that are mostly non-ASCII gibberish.
        test_docs = retriever.get_relevant_documents("fever symptoms treatment")
        if test_docs:
            sample = test_docs[0].page_content
            printable_ratio = sum(1 for c in sample if c.isascii() and c.isprintable()) / max(len(sample), 1)
            if printable_ratio < 0.7:
                log.error(
                    "[Chatbot] ❌ ChromaDB health-check FAILED — store appears corrupted "
                    "(%.0f%% printable ASCII). Delete data/chromadb/ and rebuild. "
                    "Falling back to rule-based mode.", printable_ratio * 100
                )
                return   # do NOT set _cached_retriever — forces rule-based fallback

        _cached_retriever = vs.as_retriever(search_kwargs={"k": 4})
        log.info("[Chatbot] ✅ RAG retriever loaded and cached — chat will now use medical knowledge base.")
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] RAG retriever could not be loaded: %s — chat will use rule-based mode.", exc)
    finally:
        _retriever_loading = False


def start_preload():
    """Start background preloading of the RAG retriever. Safe to call multiple times."""
    global _retriever_loading
    with _retriever_lock:
        if _cached_retriever is not None or _retriever_loading:
            return  # already loaded or loading
        _retriever_loading = True
    t = threading.Thread(target=_preload_retriever, daemon=True, name="rag-preload")
    t.start()


# Kick off preload at import time (runs in background, doesn't block)
start_preload()


# ── Ollama warm-up ────────────────────────────────────────────────────────────
# The first Ollama call loads model weights into GPU/RAM (~50s on 2GB VRAM).
# Warm-up at startup so user's first chat message doesn't wait.
_ollama_warm = False

def _warmup_ollama():
    """Send a tiny prompt to Ollama to force model loading into memory."""
    global _ollama_warm
    import requests as req
    url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    try:
        log.info("[Chatbot] Warming up Ollama (%s) — loading model into memory ...", model)
        resp = req.post(
            f"{url}/api/chat",
            json={
                "model":   model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream":  False,
                "options": {"num_predict": 1},  # generate just 1 token
            },
            timeout=120,
        )
        if resp.status_code == 200:
            _ollama_warm = True
            log.info("[Chatbot] Ollama warm-up complete — model loaded and ready.")
        else:
            log.warning("[Chatbot] Ollama warm-up got status %d.", resp.status_code)
    except req.ConnectionError:
        log.warning("[Chatbot] Ollama not reachable at %s — will retry on first chat.", url)
    except Exception as exc:
        log.warning("[Chatbot] Ollama warm-up failed: %s", exc)

threading.Thread(target=_warmup_ollama, daemon=True, name="ollama-warmup").start()


def _get_retriever(k: int = 4):
    """
    Return ChromaDB retriever if loaded, else None.
    Never blocks — if the background preload hasn't finished, returns None
    (rule-based is used for that request) and the next request will find it cached.
    """
    if _cached_retriever is not None:
        return _cached_retriever

    if _retriever_loading:
        log.info("[Chatbot] RAG still loading in background — using rule-based for this request.")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM — Ollama (primary) → Cloud API (Anthropic/OpenAI) → rule-based
# ══════════════════════════════════════════════════════════════════════════════

# Ollama config (read from env / settings.py)
_GROQ_API_KEY   = os.getenv("GROQ_API_KEY",  "")
_GROQ_MODEL     = os.getenv("GROQ_MODEL",    "llama-3.3-70b-versatile")
_GROQ_TIMEOUT   = int(os.getenv("GROQ_TIMEOUT", "30"))

_OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "gemma3:1b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # 120s for remote tunnel; local default was 90s

_SYSTEM_PROMPT_EN = (
    "You are MediSpark, a friendly medical health assistant for patients in Pakistan.\n"
    "IMPORTANT RULES:\n"
    "1. For greetings (hi, hello, salam, etc.) reply with a short friendly greeting only.\n"
    "2. For ALL medical/health questions use EXACTLY this structured format:\n\n"
    "   [One sentence describing what the symptoms likely indicate]\n\n"
    "   • [Safe home remedy or action 1]\n"
    "   • [Safe home remedy or action 2]\n"
    "   • [Dietary or lifestyle advice]\n\n"
    "   See a doctor if: [specific condition — e.g. fever persists 3+ days, symptoms worsen, etc.]\n\n"
    "   This is general health information. Please consult a qualified doctor.\n\n"
    "3. For emergencies (chest pain + sweating, blood in vomit, sudden severe headache): say GO TO EMERGENCY NOW.\n"
    "4. Never output code blocks or markdown headers (#). Keep replies under 120 words.\n"
    "5. Always use the bullet point (•) format — never write a single paragraph."
)

_SYSTEM_PROMPT_UR = (
    "You are MediSpark, a medical health assistant for patients in Pakistan.\n"
    "The user writes in Roman Urdu — Urdu language spelled with English/Latin letters, "
    "the way Pakistanis text on WhatsApp.\n\n"
    "YOUR ONLY JOB: reply ONLY in Roman Urdu using the STRUCTURED FORMAT shown below.\n\n"
    "EXAMPLE 1 — Common fever:\n"
    "User: mujhe 3 din se bukhar hai\n"
    "Reply:\n"
    "Aap ko viral infection ya flu ki wajah se bukhar ho sakta hai.\n\n"
    "• Paracetamol 500mg din mein 2 baar lein\n"
    "• Bohat sara paani aur nimbu paani piyein\n"
    "• Aaraam karein, thanda kapra mathe pe rakhen\n\n"
    "Doctor kab jayein: Agar bukhar 39°C se zyada ho ya 3 din mein theek na ho.\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "EXAMPLE 2 — Stomach complaint:\n"
    "User: pet mein dard ho raha hai khana khane ke baad\n"
    "Reply:\n"
    "Khana khane ke baad pet dard aksar gas, acidity ya indigestion ki wajah se hota hai.\n\n"
    "• Masaledar aur tailwala khana band karein\n"
    "• Adrak wali chai piyein\n"
    "• Thodi thodi der baad paani piyein\n\n"
    "Doctor kab jayein: Agar dard tez ho, ulti aaye, ya 2 din se zyada rahe.\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "EXAMPLE 3 — Cough + breathing:\n"
    "User: khansi aur saans ki takleef ho rahi hai\n"
    "Reply:\n"
    "Khansi ke sath saans ki takleef flu ya chest infection ki nishani ho sakti hai.\n\n"
    "• Shehad aur adrak wali chai piyein\n"
    "• Seene pe warm compress lagayein\n"
    "• Thanda paani aur thanday mashroobaat avoid karein\n\n"
    "Doctor kab jayein: Agar saans bohat mushkil ho ya 3 din mein theek na ho — abhi jayein.\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "EXAMPLE 4 — Diabetes signs:\n"
    "User: bohat zyada pyaas lag rahi hai aur baar baar peshab aa raha hai\n"
    "Reply:\n"
    "Zyada pyaas aur baar baar peshab aana diabetes (sugar) ki aham nishani hai.\n\n"
    "• Meetha khana aur cold drinks bilkul band karein\n"
    "• Paani zyada piyein\n"
    "• Kal hi blood sugar test karwain\n\n"
    "Doctor kab jayein: Yeh symptoms serious hain — kal hi doctor se milein.\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "EXAMPLE 5 — TB risk:\n"
    "User: mujhe 2 haftay se lagatar khansi hai aur wazan bhi kam ho raha hai\n"
    "Reply:\n"
    "2 haftay se khansi aur wazan ka kam hona TB (Tuberculosis) ya chest infection ki nishani ho sakti hai.\n\n"
    "• Paani aur ghiza zyada lein\n"
    "• Baahar zyada nikalein, fresh air lein\n"
    "• Ghar pe ilaaj se yeh theek nahi hoga\n\n"
    "Doctor kab jayein: Abhi doctor se milein — chest X-ray aur sputum test zaroor karwain.\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "EXAMPLE 6 — Emergency:\n"
    "User: achanak bohat tez sar dard hua aur ulti bhi ho rahi hai\n"
    "Reply:\n"
    "⚠️ Achanak tez sar dard aur ulti meningitis ya brain bleed ki serious nishani ho sakti hai.\n\n"
    "• Abhi hospital emergency mein jayein\n"
    "• Gaadi ya ambulance bulayen — akele mat jayein\n"
    "• Intezaar mat karein — yeh medical emergency hai\n\n"
    "Yeh sirf aam maloomat hai. Doctor se zaroor milein.\n\n"
    "ROMAN URDU VOCABULARY — ALWAYS use the RIGHT column, NEVER the WRONG column:\n"
    "  WRONG          CORRECT\n"
    "  feer           bukhar\n"
    "  fever          bukhar\n"
    "  pain           dard\n"
    "  body ki dard   jism mein dard\n"
    "  body ache      jism mein dard\n"
    "  vomiting       ulti\n"
    "  nausea         matli\n"
    "  cough          khansi\n"
    "  headache       sar dard\n"
    "  weakness       kamzori\n"
    "  dizziness      chakkar\n"
    "  diarrhea       dast\n"
    "  breathing      saans\n"
    "  sore throat    gala kharab\n"
    "  throat pain    gala dard\n"
    "  throat         gala\n"
    "  swelling       sujan\n"
    "  itching        khujli\n"
    "  treatment      ilaj\n"
    "  medicine       dawai\n"
    "  possibility    mumkin\n"
    "  week           hafta\n"
    "  water          paani\n"
    "  blood          khoon\n\n"
    "SEVERE SYMPTOMS — when user mentions any of these, say ABHI doctor/hospital jayein:\n"
    "  seene mein dard (especially with jism mein paseena ya saans ki takleef) → heart attack\n"
    "  ulti ya pakhane mein khoon → GI emergency\n"
    "  achanak bohat tez sar dard → possible stroke or brain bleed\n"
    "  bukhar ke sath confusion ya behoshi → serious infection/meningitis\n"
    "  2+ haftay khansi + wazan kam → TB possible, tests zaroori\n\n"
    "STRICT RULES — follow ALL:\n"
    "1. ONLY Roman Urdu. NEVER write full English sentences.\n"
    "2. NEVER use Arabic/Urdu script (ا ب پ ت — strictly forbidden).\n"
    "3. NEVER use 'feer', 'body ki dard', 'vomiting', 'nausea', 'cough', 'headache', "
    "'weakness', 'treatment', 'medicine', 'possibility' — use the vocabulary table above.\n"
    "4. ALWAYS use the structured format: opening sentence, then bullet points (•), "
    "then 'Doctor kab jayein:' line, then disclaimer.\n"
    "5. NEVER write a single long paragraph — always use bullet points (•) for advice.\n"
    "6. For serious symptoms: name the possible disease and recommend a specific test.\n"
    "7. Keep reply under 140 words.\n"
    "8. END every medical reply with: 'Yeh sirf aam maloomat hai. Doctor se zaroor milein.'"
)

_SYSTEM_PROMPT_UR_NATIVE = (
    "You are MediSpark, a medical health assistant for patients in Pakistan. "
    "The user is writing in Urdu script. YOU MUST reply ONLY in Urdu script (Arabic letters).\n\n"
    "STRICT RULES:\n"
    "1. صرف اردو رسم الخط میں جواب دیں۔ انگریزی بالکل نہ لکھیں۔\n"
    "2. کوڈ بلاکس یا * علامات استعمال نہ کریں۔\n"
    "3. صحت کے سوالات کے لیے: (الف) ممکنہ وجوہات، (ب) گھریلو علاج، (ج) ڈاکٹر کب جائیں۔\n"
    "4. جواب 120 الفاظ سے کم رکھیں۔\n"
    "5. ہر طبی جواب کے آخر میں لکھیں: 'یہ عام معلومات ہے۔ کسی مستند ڈاکٹر سے ضرور ملیں۔'"
)

# Legacy alias used by rule-based fallback
_SYSTEM_PROMPT = _SYSTEM_PROMPT_EN


# ── Greeting / casual interceptor ────────────────────────────────────────────
# Catches simple social phrases before they reach Ollama.
# gemma3:1b hallucinates badly when a medical system-prompt meets a casual hi.
_GREETINGS_EN = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings",
    "how are you", "how r u", "how are u", "whats up", "what's up", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "assalam", "assalamualaikum", "salam", "aoa",
}
_GREETINGS_UR = {
    "kaise", "kaisa", "kaisay",
    "kaise ho", "kaise hain", "kaisy ho", "kesy ho", "ap kaise hain",
    "theek ho", "theek hain", "kem cho", "kiddan",
    "adaab", "sat sri akal",
}
_GREETING_RESPONSE_EN = (
    "Hello! I'm MediSpark, your medical health assistant. "
    "I'm doing great, thank you for asking! 😊 "
    "How can I help you with your health today? "
    "Feel free to describe any symptoms or ask a medical question."
)
_GREETING_RESPONSE_UR = (
    "Salam! Main MediSpark hoon, aap ka sehat ka madad gar. "
    "Main bilkul theek hoon, shukriya poochne ka! 😊 "
    "Aaj main aap ki kya madad kar sakta hoon? "
    "Apne symptoms batayein ya koi bhi sehat ka sawaal poochein."
)
_GREETING_RESPONSE_UR_NATIVE = (
    "السلام علیکم! میں MediSpark ہوں، آپ کا صحت کا معاون۔ "
    "میں بالکل ٹھیک ہوں، شکریہ پوچھنے کا! 😊 "
    "آج میں آپ کی کیا مدد کر سکتا ہوں؟"
)


def _check_greeting(message: str, detected_lang: str = "english") -> str:
    """Return a language-matched greeting reply if the message is a casual greeting, else empty string."""
    clean = message.strip().lower().rstrip("!?., ")
    is_greeting = (
        clean in _GREETINGS_EN
        or clean in _GREETINGS_UR
        or any(clean == g or clean.startswith(g + " ") or clean.endswith(" " + g) for g in _GREETINGS_UR)
    )
    if not is_greeting:
        return ""
    if detected_lang == "urdu":
        return _GREETING_RESPONSE_UR_NATIVE
    if detected_lang == "roman_urdu":
        return _GREETING_RESPONSE_UR
    return _GREETING_RESPONSE_EN


def _system_prompt_for(detected_lang: str) -> str:
    if detected_lang == "urdu":
        return _SYSTEM_PROMPT_UR_NATIVE
    if detected_lang == "roman_urdu":
        return _SYSTEM_PROMPT_UR
    return _SYSTEM_PROMPT_EN


def _build_ollama_messages(
    message: str,
    history: list[dict],
    context_docs: list[str],
    detected_lang: str = "english",
) -> list[dict]:
    """
    Build the Ollama chat message list:
    [system] + optional [RAG context as system] + [history] + [user message]
    """
    messages = [{"role": "system", "content": _system_prompt_for(detected_lang)}]

    # Inject RAG context as a second system message so the LLM can reference it
    if context_docs:
        rag_text = "\n\n".join(doc[:300] for doc in context_docs[:2])
        lang_reminder = (
            " Remember: use ONLY Roman Urdu in your reply — do NOT copy English sentences from this reference."
            if detected_lang == "roman_urdu"
            else " Remember: reply ONLY in Urdu script."
            if detected_lang == "urdu"
            else ""
        )
        messages.append({
            "role": "system",
            "content": (
                "Relevant medical reference (use facts if helpful):\n\n" + rag_text + lang_reminder
            ),
        })

    # Add recent conversation history (last 4 messages = 2 exchanges)
    for entry in history[-4:]:
        role = "user" if entry["role"] == "human" else "assistant"
        messages.append({"role": role, "content": entry["content"]})

    # Current user message
    messages.append({"role": "user", "content": message})

    return messages


def _call_ollama(
    message: str,
    history: list[dict],
    context_docs: list[str],
    detected_lang: str = "english",
) -> str:
    """
    Call the local Ollama server via its /api/chat REST endpoint.
    Uses only the `requests` library (already a project dependency).
    Returns the assistant reply text, or "" on any failure.
    """
    import requests as req

    messages = _build_ollama_messages(message, history, context_docs, detected_lang)

    try:
        resp = req.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model":   _OLLAMA_MODEL,
                "messages": messages,
                "stream":  False,
                "options": {
                    "temperature":   0.3,     # lower = more focused, less hallucination
                    "num_predict":   250,     # cap tokens to prevent runaway output
                    "top_p":         0.85,
                    "top_k":         40,
                    "repeat_penalty": 1.3,   # strongly penalise repetitive/looping output
                    "stop": ["\n\n\n", "###", "```"],  # stop at code blocks / triple newlines
                },
            },
            timeout=_OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content", "").strip()
        if reply:
            log.info("[Chatbot] Ollama (%s) replied in %.1fs",
                     _OLLAMA_MODEL, data.get("total_duration", 0) / 1e9)
        return reply

    except req.ConnectionError:
        log.warning("[Chatbot] Ollama not reachable at %s — falling back.", _OLLAMA_URL)
        return ""
    except req.Timeout:
        log.warning("[Chatbot] Ollama timed out after %ds.", _OLLAMA_TIMEOUT)
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] Ollama error: %s", exc)
        return ""


def _call_groq(
    message: str,
    history: list[dict],
    context_docs: list[str],
    detected_lang: str = "english",
) -> str:
    """
    Call a cloud LLM via OpenAI-compatible REST (no extra SDK needed).

    Auto-detects provider from key prefix:
      xai-…  →  xAI Grok  (api.x.ai)    model default: grok-3-mini
      gsk_…  →  Groq       (api.groq.com) model default: llama-3.3-70b-versatile
    """
    if not _GROQ_API_KEY:
        return ""

    import requests as req

    is_xai = _GROQ_API_KEY.startswith("xai-")
    if is_xai:
        url      = "https://api.x.ai/v1/chat/completions"
        model    = _GROQ_MODEL if _GROQ_MODEL.startswith("grok") else "grok-3-mini"
        provider = "xAI Grok"
    else:
        url      = "https://api.groq.com/openai/v1/chat/completions"
        model    = _GROQ_MODEL
        provider = "Groq"

    messages = _build_ollama_messages(message, history, context_docs, detected_lang)
    try:
        resp = req.post(
            url,
            headers={
                "Authorization": f"Bearer {_GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       model,
                "messages":    messages,
                "max_tokens":  400,
                "temperature": 0.4,
                "top_p":       0.85,
            },
            timeout=_GROQ_TIMEOUT,
        )
        resp.raise_for_status()
        reply = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        if reply:
            log.info("[Chatbot] %s (%s) replied.", provider, model)
        return reply
    except req.Timeout:
        log.warning("[Chatbot] %s timed out after %ds.", provider, _GROQ_TIMEOUT)
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] %s error: %s", provider, exc)
        return ""


def _get_cloud_llm():
    """Return a LangChain LLM object (Anthropic or OpenAI), or None."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from langchain_community.llms.anthropic import Anthropic
            return Anthropic(anthropic_api_key=api_key, model="claude-instant-1.2",
                             max_tokens_to_sample=512)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Chatbot] Anthropic LLM init failed: %s", exc)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from langchain_community.llms import OpenAI
            return OpenAI(openai_api_key=openai_key, max_tokens=512)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Chatbot] OpenAI LLM init failed: %s", exc)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4. LANGUAGE ENFORCEMENT — guarantees reply matches user's input language
# ══════════════════════════════════════════════════════════════════════════════

def _is_english(text: str) -> bool:
    """
    Return True if text is English.
    Roman Urdu is written in Latin script so ASCII-ratio tests are unreliable —
    always check via detect_language() first.
    """
    try:
        from app.services.urdu_translator import detect_language
        if detect_language(text) in ("roman_urdu", "urdu"):
            return False
    except Exception:
        pass
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return True
        ascii_ratio = sum(1 for c in letters if ord(c) < 128) / len(letters)
        return ascii_ratio > 0.85


# ── Urdu script → Roman Urdu character map ───────────────────────────────────
# Used when deep-translator returns native Urdu (Arabic script) and we need
# to romanize it into Latin letters for Roman Urdu users.
_URDU_TO_ROMAN: dict[str, str] = {
    "ا": "a",  "آ": "aa", "ب": "b",  "پ": "p",  "ت": "t",  "ٹ": "T",
    "ث": "s",  "ج": "j",  "چ": "ch", "ح": "h",  "خ": "kh", "د": "d",
    "ڈ": "D",  "ذ": "z",  "ر": "r",  "ڑ": "R",  "ز": "z",  "ژ": "zh",
    "س": "s",  "ش": "sh", "ص": "s",  "ض": "z",  "ط": "t",  "ظ": "z",
    "ع": "",   "غ": "gh", "ف": "f",  "ق": "q",  "ک": "k",  "گ": "g",
    "ل": "l",  "م": "m",  "ن": "n",  "ں": "n",  "و": "o",  "ہ": "h",
    "ھ": "h",  "ء": "",   "ی": "i",  "ے": "e",  "ئ": "y",  "ؤ": "o",
    "ۃ": "h",  "لا": "la",
    # Arabic-Indic numerals
    "۰": "0",  "۱": "1",  "۲": "2",  "۳": "3",  "۴": "4",
    "۵": "5",  "۶": "6",  "۷": "7",  "۸": "8",  "۹": "9",
    # Diacritics (zabar / zer / pesh / sukun)
    "َ": "a", "ِ": "i", "ُ": "u", "ْ": "",
    "ّ": "",  "ٔ": "",  "ٕ": "",
}


def _urdu_script_to_roman(text: str) -> str:
    """Character-level transliteration: native Urdu script → Roman Urdu."""
    import re
    result = []
    i = 0
    while i < len(text):
        # Check two-character combos first (لا)
        two = text[i:i + 2]
        if two in _URDU_TO_ROMAN:
            result.append(_URDU_TO_ROMAN[two])
            i += 2
            continue
        ch = text[i]
        if ch in _URDU_TO_ROMAN:
            result.append(_URDU_TO_ROMAN[ch])
        elif ch.isascii():
            result.append(ch)
        # skip unrecognised non-ASCII characters
        i += 1
    return re.sub(r" {2,}", " ", "".join(result)).strip()


def _translate_to_roman_urdu(text: str) -> str:
    """
    Convert English medical text to Roman Urdu.

    Two-step strategy:
      Step 1 — deep-translator en→ur: get semantically correct native Urdu script.
      Step 2 — Ask Ollama to romanize the Urdu script.
                Romanizing is a much easier task for a small LLM than generating
                Roman Urdu from scratch, so output quality is far better.
      Fallback — basic character-map romanization if Ollama is offline.
    """
    import re, requests as req

    native_urdu: str = ""

    # Step 1: Translate English → native Urdu (Arabic script)
    try:
        from deep_translator import GoogleTranslator
        native_urdu = (GoogleTranslator(source="en", target="ur").translate(text[:800]) or "").strip()
    except Exception as exc:
        log.warning("[Chatbot] deep-translator en→ur failed: %s", exc)

    # Step 2: Ask Ollama to romanize the native Urdu
    if native_urdu:
        prompt = (
            "Convert the Urdu text below to Roman Urdu "
            "(write the same words using English/Latin letters, the way Pakistanis text on WhatsApp).\n\n"
            "EXAMPLE:\n"
            "Urdu:       آپ کو بخار ہے۔ پیراسیٹامول لیں اور زیادہ پانی پیئیں۔\n"
            "Roman Urdu: Aap ko bukhar hai. Paracetamol lein aur zyada paani piyein.\n\n"
            "Urdu to romanize:\n" + native_urdu[:500] + "\n\nRoman Urdu:"
        )
        try:
            resp = req.post(
                f"{_OLLAMA_URL}/api/chat",
                json={
                    "model": _OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 220, "top_p": 0.85},
                },
                timeout=45,
            )
            resp.raise_for_status()
            result = resp.json().get("message", {}).get("content", "").strip()
            result = re.sub(r"^Roman Urdu:\s*", "", result, flags=re.IGNORECASE)
            if result:
                return result
        except Exception as exc:
            log.warning("[Chatbot] Ollama romanization call failed: %s", exc)

        # Fallback: character-map romanization — only for short phrases.
        # Longer texts drop all short vowels and produce unreadable output.
        if len(native_urdu) <= 60:
            roman = _urdu_script_to_roman(native_urdu)
            if roman.strip():
                return roman

    # Nothing worked — return original text unchanged
    return text


def _translate_to_urdu_native(text: str) -> str:
    """Translate English reply to native Urdu script using deep-translator."""
    try:
        from deep_translator import GoogleTranslator
        # Translate in one chunk (cap at 4500 chars — Google's limit)
        return GoogleTranslator(source="en", target="ur").translate(text[:1000])
    except Exception as exc:
        log.warning("[Chatbot] Native Urdu translation failed: %s", exc)
        return text


def _enforce_language(reply: str, detected_lang: str) -> str:
    """
    Guarantee the reply is in the same language as the user's input.
    Only triggers translation when the LLM returned English despite instructions.

    Detection order for Roman Urdu:
      1. Use detect_language() — if it says roman_urdu the LLM did its job.
      2. If reply is native Urdu script, romanize it (LLM used wrong script).
      3. If reply is English, translate to Roman Urdu.
    """
    if detected_lang == "english" or not reply:
        return reply

    from app.services.urdu_translator import detect_language as _detect_lang

    if detected_lang == "roman_urdu":
        reply_lang = _detect_lang(reply)
        if reply_lang == "roman_urdu":
            return reply                           # LLM replied correctly — done
        if reply_lang == "urdu":
            log.info("[Chatbot] LLM replied in Urdu script for Roman Urdu input — romanizing.")
            return _urdu_script_to_roman(reply)    # convert script, no extra LLM call
        if _is_english(reply):
            log.info("[Chatbot] LLM replied in English for Roman Urdu input — translating.")
            return _translate_to_roman_urdu(reply)
        return reply                               # mixed/unknown — return as-is

    if detected_lang == "urdu":
        # Check for native Urdu script presence
        import re
        if re.search(r"[؀-ۿ]", reply):
            return reply                           # already in Urdu script
        if _is_english(reply):
            log.info("[Chatbot] LLM replied in English for Urdu input — translating.")
            return _translate_to_urdu_native(reply)
        return reply

    return reply


# ══════════════════════════════════════════════════════════════════════════════
# 5. RULE-BASED FALLBACK (no LLM key needed)
# ══════════════════════════════════════════════════════════════════════════════

_SYMPTOM_KEYWORDS = {
    "fever": "Fever can be caused by infections, inflammation, or other conditions. "
             "Stay hydrated, rest, and take paracetamol if needed. "
             "See a doctor if temperature exceeds 39°C or lasts more than 3 days.",
    "headache": "Headaches may result from tension, dehydration, or infection. "
                "Drink water, rest in a quiet dark room. "
                "Consult a doctor if severe or accompanied by stiff neck.",
    "cough": "Cough can be due to viral infections, allergies, or irritants. "
             "Stay hydrated and use honey-lemon for mild relief. "
             "Persistent cough (>2 weeks) should be evaluated by a doctor.",
    "diarrhea": "Maintain hydration with ORS. Avoid dairy and fatty foods. "
                "See a doctor if blood appears or it lasts >48 hours.",
    "vomiting": "Sip water or ORS slowly. Avoid solid food for 2–4 hours. "
                "Seek care if you cannot keep fluids down or notice blood.",
    "chest pain": "⚠️ Chest pain can indicate a serious condition such as a heart attack. "
                  "Please seek immediate medical attention — especially with sweating or breathlessness.",
    "shortness of breath": "⚠️ Difficulty breathing may require urgent care. "
                           "Please visit a doctor or emergency room immediately.",
    "breathlessness": "⚠️ Difficulty breathing may require urgent care. "
                      "Please visit a doctor or emergency room immediately.",
    "rash": "Keep the area clean and avoid scratching. "
            "Antihistamines may help for allergic rashes. "
            "See a doctor if the rash spreads or is accompanied by fever.",
    "abdominal pain": "Mild abdominal pain often resolves with rest. "
                      "Avoid spicy food. See a doctor if pain is severe or persistent.",
    "dizziness": "Sit or lie down immediately. Drink water. "
                 "Consult a doctor if dizziness is recurrent or associated with chest pain.",
    "dehydration": "Drink ORS or water frequently in small sips. "
                   "Avoid caffeine and alcohol. See a doctor if you feel confused or cannot keep fluids down.",
    "nausea": "Sip clear fluids slowly. Avoid strong smells and heavy food. "
              "See a doctor if nausea persists more than 48 hours or is accompanied by chest pain.",
    "sweating": "Excessive sweating with chest pain or breathlessness is a medical emergency. "
                "Seek immediate care. Otherwise stay hydrated and rest.",
    "fatigue": "Rest and stay hydrated. Fatigue lasting more than 2 weeks with other symptoms "
               "warrants a doctor visit to rule out anaemia, thyroid issues, or infection.",
    "diabetes": "Diabetes requires proper medical management. "
                "Monitor blood sugar, follow a low-sugar diet, exercise regularly. "
                "Consult your doctor for medication adjustments.",
    "hunger": "Excessive hunger (polyphagia) alongside thirst and frequent urination "
              "may indicate diabetes. Please consult a doctor for a blood sugar test.",
    "thirst": "Excessive thirst (polydipsia) combined with frequent urination "
              "can be a sign of diabetes or kidney issues. See a doctor promptly.",
    "urination": "Frequent urination with thirst and hunger may indicate diabetes. "
                 "A urine and blood glucose test is recommended.",
    "blood vomit": "⚠️ Vomiting blood is a medical emergency. "
                   "Go to an emergency room immediately. Do not wait.",
    "blood in vomit": "⚠️ Vomiting blood is a medical emergency. "
                      "Go to an emergency room immediately. Do not wait.",
    "vomiting blood": "⚠️ Vomiting blood is a medical emergency. "
                      "Go to an emergency room immediately. Do not wait.",
    "sudden headache": "⚠️ A sudden, severe headache (sometimes described as the worst headache "
                       "of your life) can be a sign of a brain bleed or stroke. "
                       "Seek emergency care immediately.",
    "severe headache": "⚠️ Severe headache with vomiting, fever, or neck stiffness may indicate "
                       "meningitis or another serious condition. Seek emergency care immediately.",
    "blood in stool": "⚠️ Blood in stool can indicate a serious gastrointestinal condition. "
                      "See a doctor promptly. If bleeding is heavy or you feel faint, go to emergency.",
    "weight loss": "Unexplained weight loss, especially with persistent cough or night sweats, "
                   "may indicate TB or another serious illness. "
                   "See a doctor and request a chest X-ray and sputum test.",
    "night sweats": "Night sweats combined with persistent cough or weight loss may indicate TB. "
                    "Consult a doctor and ask for a chest X-ray and sputum test.",
    "persistent cough": "A cough lasting more than 2 weeks, especially with weight loss or blood "
                        "in sputum, may indicate TB or lung disease. See a doctor immediately.",
}

# Keywords that signal the user is asking about medicines / treatment
_MEDICINE_QUESTION_KEYWORDS = [
    "medicine", "medication", "drug", "tablet", "pill", "syrup", "injection",
    "treatment", "treat", "cure", "remedy", "prescri", "antibiotic", "what to take",
    "what should i take", "what can i take", "how to treat", "dawa", "dawai",
    "ilaj", "\u0639\u0644\u0627\u062c", "\u062f\u0648\u0627",
]


def _detect_medicine_query(message: str) -> bool:
    """Return True if the user is asking about medicines or treatment."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in _MEDICINE_QUESTION_KEYWORDS)


def _find_disease_in_text(text: str) -> Optional[str]:
    """
    Try to match a disease name from MEDICINE_DB inside *text*.
    Returns the exact DB key if found, else None.
    """
    from app.services.medicine_suggester import MEDICINE_DB
    text_lower = text.lower()
    # Sort by length descending so longer (more specific) names match first
    for disease in sorted(MEDICINE_DB.keys(), key=len, reverse=True):
        if disease.lower() in text_lower:
            return disease
    return None


def _medicine_reply(disease: str) -> str:
    """Format a structured medicine recommendation card for the given disease."""
    from app.services.medicine_suggester import suggest_medicines
    meds = suggest_medicines(disease)
    otc  = meds.get("otc", [])
    rx   = meds.get("prescription", [])

    lines = [f"💊 **Recommended medications for {disease}:**", ""]

    if otc:
        lines.append("**🟢 Over-the-Counter (OTC) — available without prescription:**")
        for m in otc:
            lines.append(f"  • {m}")
        lines.append("")

    if rx:
        lines.append("**🔴 Prescription medicines (require a doctor's prescription):**")
        for m in rx:
            lines.append(f"  • {m}")
        lines.append("")

    lines.append(
        "⚠️ *This is general health information only. Do NOT self-medicate with "
        "prescription drugs. Always consult a qualified doctor before starting any treatment.*"
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5a. ROMAN URDU POST-PROCESSING CLEANUP
#     Applied AFTER LLM generation to catch English medical terms the model
#     still inserts despite glossary instructions.
# ══════════════════════════════════════════════════════════════════════════════

_RU_CLEANUPS: list[tuple[str, str]] = [
    # Fever
    (r"\bfeer\b",                                   "bukhar"),
    (r"\bfever\b",                                  "bukhar"),
    # Body / pain
    (r"\bbody\s+(?:ache|pain)\b",                   "jism mein dard"),
    (r"\bbody\s+ki\s+dard\b",                       "jism mein dard"),
    (r"\bbody\b",                                   "jism"),
    (r"\bache\b",                                   "dard"),
    (r"\bpain\b",                                   "dard"),
    # GI
    (r"\bvomiting\b",                               "ulti"),
    (r"\bnausea\b",                                 "matli"),
    (r"\bdiarrhea\b",                               "dast"),
    (r"\bdiarrhoea\b",                              "dast"),
    # Respiratory
    (r"\bcough(?:ing)?\b",                          "khansi"),
    (r"\bshortness\s+of\s+breath\b",               "saans phoolna"),
    (r"\bbreathing\s+(?:difficulty|problem|trouble|issue)\b", "saans ki takleef"),
    (r"\bdifficulty\s+(?:in\s+)?breathing\b",      "saans ki takleef"),
    (r"\bbreathing\b",                              "saans lena"),
    (r"\bbreath\b",                                 "saans"),
    # Head / throat
    (r"\bheadache\b",                               "sar dard"),
    (r"\bsore\s+throat\b",                          "gala kharab"),
    (r"\bthroat\s+(?:pain|ache|hurting|hurts)\b",  "gala dard"),
    (r"\bthroat\b",                                 "gala"),
    # Weakness / dizziness
    (r"\bweakness\b",                               "kamzori"),
    (r"\bfatigue\b",                                "thakaan"),
    (r"\bdizziness\b",                              "chakkar"),
    (r"\bdizzy\b",                                  "chakkar"),
    # Skin
    (r"\bswelling\b",                               "sujan"),
    (r"\bitching\b",                                "khujli"),
    (r"\brash\b",                                   "daane"),
    # Medicine / treatment
    (r"\bmedicines?\b",                             "dawai"),
    (r"\bmedication\b",                             "dawai"),
    (r"\btreatment\b",                              "ilaj"),
    # Vague filler
    (r"\bpossibility\b",                            "mumkin"),
    (r"\bpossible\b",                               "mumkin"),
    # Time / units
    (r"\bweeks\b",                                  "haftay"),
    (r"\bweek\b",                                   "hafta"),
    (r"\bwater\b",                                  "paani"),
    (r"\bstomach\b",                                "pet"),
]


def _fix_roman_urdu_response(text: str) -> str:
    """
    Replace English medical terms that slip into LLM Roman Urdu output
    with their proper Roman Urdu equivalents.
    Only called when detected_lang == 'roman_urdu'.
    """
    import re
    for pat, repl in _RU_CLEANUPS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r" {2,}", " ", text)
    return text


_URGENT_COMBO = {
    # Cardiac emergencies
    frozenset({"chest pain", "sweating"}),
    frozenset({"chest pain", "breathlessness"}),
    frozenset({"chest pain", "nausea"}),
    frozenset({"chest pain", "dizziness"}),
    frozenset({"chest pain", "sweating", "breathlessness"}),
    # GI bleeding
    frozenset({"blood", "vomiting"}),
    frozenset({"blood", "stool"}),
    # Neurological (potential stroke / meningitis)
    frozenset({"sudden", "headache", "vomiting"}),
    frozenset({"severe headache", "confusion"}),
    frozenset({"severe headache", "fever"}),
}

_URGENT_REPLY = (
    "⚠️ **This is a medical emergency.**\n\n"
    "The combination of symptoms you described — chest pain, breathlessness, sweating, and/or nausea — "
    "are classic warning signs of a **heart attack** or other serious cardiac event.\n\n"
    "**Please call emergency services (115) or go to the nearest emergency room immediately.**\n\n"
    "Do NOT wait. Do NOT drive yourself. Every minute matters.\n\n"
    "⚠️ *This is urgent health guidance. Seek immediate professional medical care.*"
)


def _rule_based_reply(message: str, context_docs: list[str], history: Optional[list] = None) -> str:
    """Generate a helpful rule-based reply when no LLM is available."""
    # Normalize underscores → spaces so chest_pain matches "chest pain"
    msg_lower = message.lower().replace("_", " ")

    # ── Emergency combination check (before individual keywords) ─────────────
    present = {kw for kw in _SYMPTOM_KEYWORDS if kw in msg_lower}
    for combo in _URGENT_COMBO:
        if combo.issubset(present):
            return _URGENT_REPLY

    # ── Medicine / treatment question? ────────────────────────────────────────
    if _detect_medicine_query(message):
        # 1. Try to find a disease name in the current message
        disease = _find_disease_in_text(message)

        # 2. Fall back: scan the last few AI messages in history for a disease
        if not disease and history:
            for entry in reversed(history[-10:]):
                if entry.get("role") == "ai":
                    disease = _find_disease_in_text(entry.get("content", ""))
                    if disease:
                        break

        # 3. Try to find a disease in RAG context docs
        if not disease:
            for doc in context_docs:
                disease = _find_disease_in_text(doc)
                if disease:
                    break

        if disease:
            return _medicine_reply(disease)
        else:
            return (
                "I can provide medication information once I know the specific disease. "
                "Please first describe your symptoms using the **Symptom Checker**, "
                "or mention the disease name directly (e.g. 'medicine for typhoid').\n\n"
                "⚠️ *Always consult a qualified doctor before taking any medication.*"
            )

    # ── Symptom keywords ──────────────────────────────────────────────────────
    for keyword, response in _SYMPTOM_KEYWORDS.items():
        if keyword in msg_lower or keyword.replace(" ", "_") in message.lower():
            reply = f"**{keyword.title()} guidance:**\n{response}"
            if context_docs:
                reply += f"\n\n📚 *From medical knowledge base:*\n{context_docs[0][:300]}..."
            reply += "\n\n⚠️ *This is general health information, not medical advice. Please consult a qualified doctor.*"
            return reply

    # ── Generic response with RAG context ────────────────────────────────────
    if context_docs:
        return (
            f"Based on your query, here is relevant medical information:\n\n"
            f"{context_docs[0][:400]}\n\n"
            f"⚠️ *This is general health information. Please consult a qualified doctor.*"
        )

    return (
        "I'm your MediSpark health assistant. I can help with symptom information, "
        "disease queries, medicine recommendations, and general health guidance. "
        "Describe your symptoms or ask about a specific disease or medication.\n\n"
        "⚠️ *This is general health information, not a substitute for professional medical advice.*"
    )


# (Section 5 removed — LangChain ConversationalRetrievalChain replaced by
#  direct Ollama /api/chat calls which handle multi-turn natively.)


# ══════════════════════════════════════════════════════════════════════════════
# 6. KAFKA PUBLISH
# ══════════════════════════════════════════════════════════════════════════════

def _publish_to_kafka(user_id: str, message: str, reply: str) -> None:
    try:
        from app.services.kafka_producer import publish
        publish(
            "chat-messages",
            {
                "user_id":  user_id,
                "message":  message,
                "reply":    reply,
            },
            key=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] Kafka publish failed (non-fatal): %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 7. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def chat(user_id: str, message: str) -> dict:
    """
    Process a chat message from *user_id* and return a response dict.

    LLM priority chain:
        1. Ollama (local, free, no API key)  ← gemma3:1b
        2. Anthropic Claude / OpenAI         ← cloud fallback
        3. Rule-based fallback               ← always works

    Returns
    -------
    {
        "reply":          str,
        "detected_lang":  str,        # 'english' | 'urdu' | 'roman_urdu'
        "translated":     str | None, # English translation (if input wasn't English)
        "sources":        list[str],  # RAG source snippets used
        "llm_source":     str,        # 'ollama' | 'anthropic' | 'openai' | 'rule-based'
    }
    """
    from app.services.urdu_translator import detect_language, translate_to_english

    # 1. Language detection + translation
    detected_lang = detect_language(message)
    if detected_lang != "english":
        translated_msg = translate_to_english(message)
        log.info("[Chatbot] Translated '%s' -> '%s'", message[:60], translated_msg[:60])
    else:
        translated_msg = message

    # 2. Load conversation history
    history = _load_history(user_id)

    # 3a. Greeting short-circuit — intercept before hitting the LLM
    greeting_reply = _check_greeting(message, detected_lang)
    if not greeting_reply:
        greeting_reply = _check_greeting(translated_msg, detected_lang)
    if greeting_reply:
        history.append({"role": "human", "content": message})
        history.append({"role": "ai",    "content": greeting_reply})
        _save_history(user_id, history)
        _publish_to_kafka(user_id, message, greeting_reply)
        return {
            "reply":         greeting_reply,
            "detected_lang": detected_lang,
            "translated":    translated_msg if detected_lang != "english" else None,
            "sources":       [],
            "llm_source":    "greeting-interceptor",
        }

    # 3b. Retrieve RAG context (non-blocking)
    context_docs: list[str] = []
    retriever = _get_retriever(k=4)
    if retriever:
        try:
            docs = retriever.get_relevant_documents(translated_msg)
            # Sanitize: keep only printable ASCII to prevent garbage from
            # corrupted ChromaDB chunks confusing the small LLM
            raw_docs = [d.page_content for d in docs]
            context_docs = [
                "".join(c for c in doc if c.isprintable())[:300]
                for doc in raw_docs
                if doc.strip()
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("[Chatbot] Retriever error: %s", exc)

    # 4. Generate reply — LLM priority chain
    reply = ""
    llm_source = "rule-based"

    # For Roman Urdu / Urdu inputs send the English translation to the LLM.
    # The language-specific system prompt instructs it to REPLY in the user's language.
    llm_input = translated_msg if detected_lang in ("roman_urdu", "urdu") else message

    # 4a. Try Groq cloud API first (fast, free, reliable — no local setup needed)
    if not reply:
        groq_reply = _call_groq(llm_input, history, context_docs, detected_lang)
        if groq_reply:
            reply = groq_reply
            llm_source = f"groq/{_GROQ_MODEL}"

    # 4b. Fall back to local Ollama if Groq key not set or Groq failed
    if not reply:
        ollama_reply = _call_ollama(llm_input, history, context_docs, detected_lang)
        if ollama_reply:
            reply = ollama_reply
            llm_source = f"ollama/{_OLLAMA_MODEL}"

    # 4c. Try cloud LLM (Anthropic / OpenAI) if both above failed
    if not reply:
        cloud_llm = _get_cloud_llm()
        if cloud_llm and retriever:
            llm_result = [""]
            def _llm_call():
                try:
                    from langchain.memory import ConversationBufferMemory
                    from langchain.chains import ConversationalRetrievalChain
                    memory = ConversationBufferMemory(
                        memory_key="chat_history", return_messages=True, output_key="answer"
                    )
                    for entry in history[-10:]:
                        if entry["role"] == "human":
                            memory.chat_memory.add_user_message(entry["content"])
                        else:
                            memory.chat_memory.add_ai_message(entry["content"])
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=cloud_llm, retriever=retriever, memory=memory,
                        return_source_documents=False, verbose=False,
                    )
                    result = chain({"question": f"{_SYSTEM_PROMPT}\n\nUser: {translated_msg}"})
                    llm_result[0] = result.get("answer", "").strip()
                except Exception as exc:
                    log.error("[Chatbot] Cloud LLM error: %s", exc)

            llm_thread = threading.Thread(target=_llm_call, daemon=True)
            llm_thread.start()
            llm_thread.join(timeout=_LLM_TIMEOUT)
            if llm_result[0]:
                reply = llm_result[0]
                llm_source = "cloud-llm"

    # 4d. Final fallback — rule-based (instant, no network)
    if not reply:
        reply = _rule_based_reply(translated_msg, context_docs, history)
        llm_source = "rule-based"

    # 4e. Language enforcement — translate reply if LLM ignored language instruction.
    # Rule-based responses stay in English: character-map romanisation (Ollama fallback)
    # produces garbled output for paragraph-length text, so English is more readable.
    if llm_source != "rule-based" and reply != _URGENT_REPLY:
        reply = _enforce_language(reply, detected_lang)

    # 4f. Roman Urdu post-processing — replace stray English medical terms
    if detected_lang == "roman_urdu" and llm_source != "rule-based":
        reply = _fix_roman_urdu_response(reply)

    # 5. Update history
    history.append({"role": "human", "content": message})
    history.append({"role": "ai",    "content": reply})
    _save_history(user_id, history)

    # 6. Publish to Kafka (non-fatal)
    _publish_to_kafka(user_id, message, reply)

    return {
        "reply":         reply,
        "detected_lang": detected_lang,
        "translated":    translated_msg if detected_lang != "english" else None,
        "sources":       context_docs[:2],
        "llm_source":    llm_source,
    }


def reset_memory(user_id: str) -> None:
    """Clear the conversation history for *user_id*."""
    _save_history(user_id, [])
    _in_process_memory.pop(user_id, None)
    log.info("[Chatbot] Conversation memory reset for user %s", user_id)


def get_history(user_id: str) -> list[dict]:
    """Return the full conversation history for *user_id*."""
    return _load_history(user_id)


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("MediSpark Chatbot — Smoke Test (rule-based, no LLM key needed)")
    print("=" * 60)

    test_messages = [
        ("user-001", "I have fever and headache"),
        ("user-001", "How long should I rest?"),       # tests multi-turn
        ("user-001", "mujhe khansi bhi ho rahi hai"),  # Roman Urdu
    ]

    for uid, msg in test_messages:
        print(f"\n[User] {msg}")
        result = chat(uid, msg)
        print(f"[Lang]  {result['detected_lang']}")
        if result["translated"]:
            print(f"[Trans] {result['translated']}")
        print(f"[Bot]   {result['reply'][:200]}...")
