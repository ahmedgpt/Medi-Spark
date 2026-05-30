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
from typing import Optional

log = logging.getLogger(__name__)

# Redis key pattern for conversation history
_REDIS_KEY_PREFIX = "medispark:chat_history:"
_MAX_HISTORY = 20       # messages to keep per user (10 exchanges)


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
# 2. RAG RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def _get_retriever(k: int = 4):
    """Return ChromaDB retriever if vector store is available, else None."""
    try:
        from app.services.rag_engine import build_vectorstore
        vs = build_vectorstore()
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception as exc:  # noqa: BLE001
        log.warning("[Chatbot] RAG retriever unavailable: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM — Anthropic Claude (primary) → rule-based fallback
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm():
    """Return a LangChain LLM object, or None if no API key is configured."""
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

    return None   # will trigger rule-based fallback


# ══════════════════════════════════════════════════════════════════════════════
# 4. RULE-BASED FALLBACK (no LLM key needed)
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


def _rule_based_reply(message: str, context_docs: list[str]) -> str:
    """Generate a helpful rule-based reply when no LLM is available."""
    msg_lower = message.lower()

    # Check for known symptom keywords
    for keyword, response in _SYMPTOM_KEYWORDS.items():
        if keyword in msg_lower:
            reply = f"**{keyword.title()} guidance:**\n{response}"
            if context_docs:
                reply += f"\n\n📚 *From medical knowledge base:*\n{context_docs[0][:300]}..."
            reply += "\n\n⚠️ *This is general health information, not medical advice. Please consult a qualified doctor.*"
            return reply

    # Generic response with RAG context
    if context_docs:
        return (
            f"Based on your query, here is relevant medical information:\n\n"
            f"{context_docs[0][:400]}\n\n"
            f"⚠️ *This is general health information. Please consult a qualified doctor.*"
        )

    return (
        "I'm your MediSpark health assistant. I can help with symptom information, "
        "disease queries, and general health guidance. Please describe your symptoms "
        "and I'll provide relevant medical information.\n\n"
        "⚠️ *This is general health information, not a substitute for professional medical advice.*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. LANGCHAIN CONVERSATIONAL CHAIN
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are MediSpark, an intelligent medical assistant for patients in Pakistan.
You provide clear, accurate health information in simple language.
When asked about symptoms, provide:
1. Likely causes
2. When to see a doctor (urgency)
3. Safe home remedies if appropriate
4. Recommended tests if relevant

Always end with: "⚠️ This is general health information. Please consult a qualified doctor for diagnosis and treatment."
You understand both English and Roman Urdu medical terms."""


def _build_langchain_reply(
    message: str,
    history: list[dict],
    llm,
    retriever,
) -> str:
    """Build reply using LangChain ConversationalRetrievalChain."""
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
        from langchain.prompts import PromptTemplate

        # Rebuild in-memory LangChain memory from stored history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        for entry in history[-10:]:  # last 5 exchanges
            if entry["role"] == "human":
                memory.chat_memory.add_user_message(entry["content"])
            else:
                memory.chat_memory.add_ai_message(entry["content"])

        # Build chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False,
        )

        result = chain({"question": f"{_SYSTEM_PROMPT}\n\nUser: {message}"})
        return result.get("answer", "").strip()

    except Exception as exc:  # noqa: BLE001
        log.error("[Chatbot] LangChain chain error: %s", exc)
        return ""


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

    Returns
    -------
    {
        "reply":          str,
        "detected_lang":  str,        # 'english' | 'urdu' | 'roman_urdu'
        "translated":     str | None, # English translation (if input wasn't English)
        "sources":        list[str],  # RAG source snippets used
    }
    """
    from app.services.urdu_translator import detect_language, translate_to_english

    # 1. Language detection + translation
    detected_lang = detect_language(message)
    if detected_lang != "english":
        translated_msg = translate_to_english(message)
        log.info("[Chatbot] Translated '%s' → '%s'", message[:60], translated_msg[:60])
    else:
        translated_msg = message

    # 2. Load conversation history
    history = _load_history(user_id)

    # 3. Retrieve RAG context
    context_docs: list[str] = []
    retriever = _get_retriever(k=4)
    if retriever:
        try:
            docs = retriever.get_relevant_documents(translated_msg)
            context_docs = [d.page_content[:300] for d in docs]
        except Exception as exc:  # noqa: BLE001
            log.warning("[Chatbot] Retriever error: %s", exc)

    # 4. Generate reply
    llm = _get_llm()
    reply = ""

    if llm and retriever:
        reply = _build_langchain_reply(translated_msg, history, llm, retriever)

    if not reply:
        # Rule-based fallback (works without any API key)
        reply = _rule_based_reply(translated_msg, context_docs)

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
        "sources":       context_docs[:2],   # return top-2 snippets to UI
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
