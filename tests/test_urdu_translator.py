"""Tests for Urdu translator / language detection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.urdu_translator import (
    detect_language,
    translate_to_english,
    normalize_medical_terms,
    _apply_medical_dict,
)


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("I have fever and cough") == "english"

    def test_english_complex(self):
        assert detect_language("I'm suffering from a severe headache with sore eyes") == "english"

    def test_roman_urdu_multiple_hints(self):
        assert detect_language("mujhe bukhar aur sar dard hai") == "roman_urdu"

    def test_roman_urdu_strong_hint(self):
        assert detect_language("bukhar") == "roman_urdu"

    def test_native_urdu(self):
        assert detect_language("بخار اور کھانسی ہے") == "urdu"

    def test_empty_string(self):
        assert detect_language("") == "english"

    def test_whitespace(self):
        assert detect_language("   ") == "english"


class TestMedicalDict:
    def test_bukhar(self):
        result = _apply_medical_dict("bukhar")
        assert "fever" in result

    def test_sar_dard(self):
        result = _apply_medical_dict("sar dard")
        assert "headache" in result

    def test_multi_word_priority(self):
        """Longer phrases like 'tez bukhar' should match before 'bukhar'."""
        result = _apply_medical_dict("tez bukhar")
        assert "high fever" in result

    def test_khansi(self):
        result = _apply_medical_dict("khansi")
        assert "cough" in result


class TestTranslateToEnglish:
    def test_english_passthrough(self):
        msg = "I have a headache"
        assert translate_to_english(msg) == msg

    def test_roman_urdu_dict_only(self):
        """When deep-translator is not available, dict substitution still works."""
        result = translate_to_english("bukhar aur khansi")
        assert "fever" in result.lower()
        assert "cough" in result.lower()

    def test_empty_input(self):
        assert translate_to_english("") == ""


class TestNormalizeMedicalTerms:
    def test_stomachache(self):
        assert "abdominal pain" in normalize_medical_terms("stomachache")

    def test_loose_motions(self):
        assert "diarrhea" in normalize_medical_terms("loose motions")

    def test_passthrough(self):
        assert normalize_medical_terms("fever") == "fever"
