"""
Day 17-18: Roman Urdu / English NLP
=====================================
Supports three input modes:
  1. Plain English          → passed through unchanged
  2. Roman Urdu (Romanised) → mapped through medical dictionary + Google Translate
  3. Native Urdu (Unicode)  → Google Translate directly

Usage
-----
    from app.services.urdu_translator import translate_to_english, detect_language

    text = "mujhe bukhar aur sar dard hai"
    lang = detect_language(text)           # → "roman_urdu"
    eng  = translate_to_english(text)      # → "I have fever and headache"
"""
from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. ROMAN URDU → ENGLISH MEDICAL DICTIONARY
#    Coverage: common symptoms expressed in everyday Pakistani Roman Urdu.
#    Keys are lower-cased Roman Urdu words/phrases; values are English equivalents.
# ══════════════════════════════════════════════════════════════════════════════
ROMAN_URDU_MEDICAL_DICT: dict[str, str] = {
    # ── Body / General ────────────────────────────────────────────────────────
    "bukhar":           "fever",
    "bukhaar":          "fever",
    "tez bukhar":       "high fever",
    "sar dard":         "headache",
    "sardard":          "headache",
    "sar ka dard":      "headache",
    "dard":             "pain",
    "takleef":          "pain",
    "jism dard":        "body ache",
    "badan dard":       "body ache",
    "dard ho raha hai": "I am in pain",
    "kamzori":          "weakness",
    "thakaan":          "fatigue",
    "chakkar":          "dizziness",
    "chakkar aana":     "dizziness",
    "behoshi":          "loss of consciousness",
    "gardan dard":      "neck pain",
    "kamar dard":       "back pain",

    # ── Respiratory ───────────────────────────────────────────────────────────
    "khansi":           "cough",
    "khasi":            "cough",
    "saans lena mushkil": "difficulty breathing",
    "saans phoolna":    "shortness of breath",
    "naak bagna":       "runny nose",
    "naak band":        "blocked nose",
    "nazla":            "cold / nasal congestion",
    "zukam":            "cold",
    "gale mein dard":   "sore throat",
    "gala kharab":      "sore throat",

    # ── Gastro ───────────────────────────────────────────────────────────────
    "ulti":             "vomiting",
    "qay":              "vomiting",
    "matli":            "nausea",
    "dast":             "diarrhea",
    "qabz":             "constipation",
    "pait dard":        "abdominal pain",
    "pet dard":         "abdominal pain",
    "bhook nahi":       "loss of appetite",
    "bhook na lagna":   "loss of appetite",

    # ── Skin ─────────────────────────────────────────────────────────────────
    "khujli":           "itching",
    "daane":            "rash",
    "dane":             "rash",
    "sujan":            "swelling",
    "soraa":            "wound / sore",

    # ── Chest / Cardiac ───────────────────────────────────────────────────────
    "seene mein dard":  "chest pain",
    "sine mein dard":   "chest pain",
    "dil ki dharkan":   "palpitations",
    "dil tez dhadakna": "rapid heartbeat",

    # ── Urinary ──────────────────────────────────────────────────────────────
    "peshab mein jalan": "burning urination",
    "bar bar peshab":   "frequent urination",

    # ── Neuro ─────────────────────────────────────────────────────────────────
    "aankhon ke aage andhera": "blurred vision",
    "yaad nahi rehta": "memory loss",
    "haath pair sonne":  "numbness in hands and feet",

    # ── Duration / Time ──────────────────────────────────────────────────────
    "kal se":           "since yesterday",
    "kuch dino se":     "for a few days",
    "haftay se":        "for a week",

    # ── Severity / Qualifiers ─────────────────────────────────────────────────
    "bohat":            "very",
    "thoda":            "a little",
    "zyada":            "more",
    "kam":              "less",
    "acha nahi feel":   "not feeling well",
    "tabiyat theek nahi": "not feeling well",
    "tabiyat kharab":   "feeling unwell",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Urdu Unicode block: U+0600 – U+06FF
_URDU_UNICODE_RE = re.compile(r"[\u0600-\u06FF]")

# High-frequency Roman Urdu function words / symptom roots
_ROMAN_URDU_HINTS = {
    # Medical terms
    "bukhar", "bukhaar", "dard", "khansi", "khasi", "matli", "ulti", "qay", "dast",
    "kamzori", "thakaan", "chakkar", "sujan", "khujli", "daane", "dane",
    "zukam", "nazla", "saans", "tabiyat", "bohat", "bahut", "zyada", "mujhe",
    "seene", "sine", "peshab", "peet", "gardan", "kamar", "jism", "badan",
    # Common verbs / copulas
    "hai", "hain", "ho", "raha", "rahi", "hoon", "hun", "tha", "thi",
    "hoga", "hogi", "hua", "hui", "jayein", "jayen", "karein", "karen",
    "lein", "len", "piyein", "piyen",
    # Pronouns
    "ye", "yeh", "wo", "woh", "main", "mein", "aap", "ap", "hum", "tum",
    # Question words
    "kya", "kyun", "kab", "kahan", "kaun", "kis", "kaise", "kitnay",
    # Particles / postpositions
    "se", "ka", "ki", "ke", "ko", "par", "mein", "tak", "bhi",
    "toh", "to", "aur", "ya", "nahi", "nahin", "na", "phir", "ab",
    # Common short forms used in Pakistani texting
    "sb", "sab", "k", "h", "hn", "ap", "kch", "kuch", "btao", "batao",
    # Time words
    "kal", "aj", "aaj", "abhi", "parso",
    # Very common Roman Urdu words in medical replies
    "salaam", "salam",
    "ghar", "khana", "khaana",
    "paani", "pani",
    "theek", "thik",
    "zaroor",
    "wajah", "waja",
    "ilaj",
    "doctor", # used in Roman Urdu responses constantly
    "sirf", "aam", "maloomat",
    "aksar", "mushkil",
    "shehad", "adrak",
    "masla",
    "nishani", "nishaan",
    "shukriya", "shukr",
    "takleef",
    "apka", "aapka", "apki", "aapki",
}

# Punctuation stripper for tokenisation
_PUNCT_RE = re.compile(r"[^\w\s]")


def detect_language(text: str) -> str:
    """
    Returns one of:
      'urdu'       — native Urdu script (Unicode)
      'roman_urdu' — Roman/transliterated Urdu
      'english'    — English
    """
    if not text or not text.strip():
        return "english"

    # Native Urdu script detection (fast path)
    if _URDU_UNICODE_RE.search(text):
        return "urdu"

    # Strip punctuation before tokenising so "hain?" matches "hain"
    clean = _PUNCT_RE.sub(" ", text.lower())
    tokens = set(clean.split())

    # Need at least 2 hint matches to avoid false positives on English
    overlap = tokens & _ROMAN_URDU_HINTS
    if len(overlap) >= 2:
        return "roman_urdu"

    # Single strong medical term is enough on its own
    _strong_hints = {
        "bukhar", "dard", "khansi", "matli", "ulti", "qay", "dast",
        "kamzori", "thakaan", "chakkar", "sujan", "khujli", "daane",
        "zukam", "nazla", "saans", "tabiyat", "mujhe",
    }
    if tokens & _strong_hints:
        return "roman_urdu"

    return "english"


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROMAN URDU → ENGLISH (dictionary pass)
# ══════════════════════════════════════════════════════════════════════════════

def _apply_medical_dict(text: str) -> str:
    """
    Replace known Roman Urdu medical phrases with English equivalents.
    Longer phrases are matched first to avoid partial substitutions.
    """
    text_lower = text.lower()
    # Sort by descending phrase length so multi-word phrases match first
    for roman, english in sorted(ROMAN_URDU_MEDICAL_DICT.items(), key=lambda x: -len(x[0])):
        text_lower = text_lower.replace(roman, english)
    return text_lower


# ══════════════════════════════════════════════════════════════════════════════
# 4. GOOGLE TRANSLATE VIA deep-translator
# ══════════════════════════════════════════════════════════════════════════════

def _google_translate(text: str, source: str = "auto") -> str:
    """Translate *text* to English using deep-translator (Google backend)."""
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source, target="en").translate(text)
        return translated or text
    except Exception as exc:  # noqa: BLE001
        log.warning("[Translator] Google Translate failed: %s", exc)
        return text


# ══════════════════════════════════════════════════════════════════════════════
# 5. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def translate_to_english(text: str) -> str:
    """
    Accepts any of: English, native Urdu, Roman Urdu.
    Returns an English string suitable for symptom extraction.

    Pipeline:
      1. Detect language.
      2. If native Urdu   → Google Translate (ur→en).
      3. If Roman Urdu    → medical dict substitution → Google Translate (auto→en).
      4. If English       → return as-is.
    """
    if not text or not text.strip():
        return text

    lang = detect_language(text)
    log.debug("[Translator] Detected language: %s | Input: %s", lang, text[:80])

    if lang == "urdu":
        translated = _google_translate(text, source="ur")
        log.info("[Translator] Urdu→English: %s → %s", text[:60], translated[:60])
        return translated

    if lang == "roman_urdu":
        # Run Google Translate on the ORIGINAL text first.
        # Google handles Pakistani Roman Urdu (source=auto) reliably, and applying
        # the medical dict BEFORE Google creates hybrid text like "aik haftay se cough hai"
        # which Google then mis-parses ("aik for a week cough").
        translated = _google_translate(text, source="auto")
        if translated and translated.lower().strip() != text.lower().strip():
            log.info("[Translator] Roman Urdu→English (Google direct): %s → %s",
                     text[:60], translated[:60])
            return translated
        # Fallback only if Google returned the same string unchanged: apply medical dict
        after_dict = _apply_medical_dict(text)
        translated = _google_translate(after_dict, source="auto")
        log.info("[Translator] Roman Urdu→English (dict+Google): %s → %s",
                 text[:60], translated[:60])
        return translated

    # English — pass through
    return text


def normalize_medical_terms(text: str) -> str:
    """
    Optionally normalise synonyms *after* translation, e.g.
    'stomachache' → 'abdominal pain', 'loose motions' → 'diarrhea'.
    Extend this dict as needed.
    """
    _synonyms = {
        "stomachache":      "abdominal pain",
        "stomach ache":     "abdominal pain",
        "tummy ache":       "abdominal pain",
        "loose motions":    "diarrhea",
        "loose stool":      "diarrhea",
        "running nose":     "runny nose",
        "breathlessness":   "shortness of breath",
        "throwing up":      "vomiting",
        "throwing-up":      "vomiting",
        "feel sick":        "nausea",
        "feeling sick":     "nausea",
    }
    text_lower = text.lower()
    for syn, canonical in _synonyms.items():
        text_lower = text_lower.replace(syn, canonical)
    return text_lower


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    samples = [
        "mujhe bukhar aur sar dard hai",
        "I have fever and cough",
        "بخار اور کھانسی ہے",
        "khansi aur saans lena mushkil ho raha hai",
        "seene mein dard aur chakkar aa raha hai",
    ]
    print("=" * 60)
    print("Urdu Translator — Smoke Test")
    print("=" * 60)
    for s in samples:
        lang = detect_language(s)
        eng  = translate_to_english(s)
        norm = normalize_medical_terms(eng)
        print(f"\nInput   : {s}")
        print(f"Lang    : {lang}")
        print(f"English : {eng}")
        print(f"Normal  : {norm}")
