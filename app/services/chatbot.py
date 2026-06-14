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
_OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
_OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "gemma3:1b")
_OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # 120s for remote tunnel; local default was 90s

_SYSTEM_PROMPT_EN = (
    "You are MediSpark, a friendly medical health assistant for patients in Pakistan. "
    "IMPORTANT RULES:\n"
    "1. Reply in plain English. Never output code blocks.\n"
    "2. If the user greets you (e.g. 'hello', 'hi', 'how are you'), respond with a short friendly greeting and ask how you can help with their health today.\n"
    "3. For health/medical questions: give (a) likely causes, (b) safe home remedies, (c) when to see a doctor.\n"
    "4. Keep every reply under 150 words.\n"
    "5. End medical replies with: This is general health information. Please consult a qualified doctor."
)

_SYSTEM_PROMPT_UR = (
    "You are MediSpark, a medical health assistant for patients in Pakistan. "
    "The user is writing in Roman Urdu — Urdu language spelled with English/Latin letters.\n\n"
    "YOUR ONLY JOB: reply in Roman Urdu. Here is an example of correct Roman Urdu:\n"
    "  'Aap ka masla samajh aa gaya. Bukhar ke liye paracetamol lein aur zyada paani piyein. "
    "Agar teen din mein theek na hon toh doctor se zaroor milein.'\n\n"
    "STRICT RULES — follow all of them:\n"
    "1. ONLY write in Roman Urdu. NEVER switch to English sentences.\n"
    "2. NEVER use Arabic/Urdu script (no ا ب پ letters).\n"
    "3. NEVER output code blocks or bullet symbols like *.\n"
    "4. For greetings: reply with a short warm Roman Urdu greeting.\n"
    "5. For health questions: (a) wajuhaat batayein, (b) ghar pe ilaj, (c) doctor kab jayein.\n"
    "6. Keep reply under 120 words.\n"
    "7. End every medical reply with: 'Yeh sirf aam maloomat hai. Doctor se zaroor milein.'"
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
        messages.append({
            "role": "system",
            "content": (
                "Relevant medical reference (use if helpful):\n\n" + rag_text
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
    """Return True if text appears to be primarily English."""
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        # Fallback: if >60% ASCII letters it's likely English/Roman
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return True
        ascii_ratio = sum(1 for c in letters if ord(c) < 128) / len(letters)
        return ascii_ratio > 0.85


def _translate_to_roman_urdu(text: str) -> str:
    """Ask the LLM to rewrite an English reply in Roman Urdu."""
    import requests as req
    prompt = (
        "Translate the following medical advice into Roman Urdu "
        "(Urdu language written with English/Latin letters, the way Pakistanis text each other). "
        "Example style: 'Aap ko bukhar hai. Paracetamol lein aur zyada paani piyein. "
        "Teen din mein theek na hon toh doctor se milein.' "
        "Output ONLY the Roman Urdu translation — no English, no Urdu script:\n\n" + text[:600]
    )
    try:
        resp = req.post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model": _OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 220, "top_p": 0.9},
            },
            timeout=45,
        )
        resp.raise_for_status()
        result = resp.json().get("message", {}).get("content", "").strip()
        return result if result else text
    except Exception as exc:
        log.warning("[Chatbot] Roman Urdu translation call failed: %s", exc)
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
    Only triggers a translation when the LLM returned English despite instructions.
    """
    if detected_lang == "english" or not reply:
        return reply

    if not _is_english(reply):
        return reply   # LLM already replied in the right language — nothing to do

    if detected_lang == "roman_urdu":
        log.info("[Chatbot] LLM replied in English for Roman Urdu input — translating.")
        return _translate_to_roman_urdu(reply)

    if detected_lang == "urdu":
        log.info("[Chatbot] LLM replied in English for Urdu input — translating.")
        return _translate_to_urdu_native(reply)

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
    "chest pain": "⚠️ Chest pain can indicate a serious condition. "
                  "Please seek immediate medical attention.",
    "shortness of breath": "⚠️ Difficulty breathing may require urgent care. "
                           "Please visit a doctor or emergency room immediately.",
    "rash": "Keep the area clean and avoid scratching. "
            "Antihistamines may help for allergic rashes. "
            "See a doctor if the rash spreads or is accompanied by fever.",
    "abdominal pain": "Mild abdominal pain often resolves with rest. "
                      "Avoid spicy food. See a doctor if pain is severe or persistent.",
    "dizziness": "Sit or lie down immediately. Drink water. "
                 "Consult a doctor if dizziness is recurrent or associated with chest pain.",
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


def _rule_based_reply(message: str, context_docs: list[str], history: Optional[list] = None) -> str:
    """Generate a helpful rule-based reply when no LLM is available."""
    msg_lower = message.lower()

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
        if keyword in msg_lower:
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

    # 4a. Try Ollama (local LLM) first
    # Send the ORIGINAL message so the model can match the user's language.
    # The system prompt instructs it to reply in the same language.
    if not reply:
        ollama_reply = _call_ollama(message, history, context_docs, detected_lang)
        if ollama_reply:
            reply = ollama_reply
            llm_source = f"ollama/{_OLLAMA_MODEL}"

    # 4b. Try cloud LLM (Anthropic / OpenAI) if Ollama failed
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

    # 4c. Final fallback — rule-based (instant, no network)
    if not reply:
        reply = _rule_based_reply(translated_msg, context_docs, history)
        llm_source = "rule-based"

    # 4d. Language enforcement — translate reply if LLM ignored language instruction
    if llm_source != "rule-based":
        reply = _enforce_language(reply, detected_lang)

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
