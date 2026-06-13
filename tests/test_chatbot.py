"""Tests for chatbot service (rule-based mode — no LLM key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chatbot import chat, get_history, reset_memory


class TestChatRuleBased:
    """Test the rule-based chatbot fallback (works without LLM API keys)."""

    USER_ID = "test-user-001"

    def setup_method(self):
        reset_memory(self.USER_ID)

    def test_returns_dict(self):
        result = chat(self.USER_ID, "I have fever")
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = chat(self.USER_ID, "I have a cough")
        assert "reply" in result
        assert "detected_lang" in result
        assert "sources" in result

    def test_fever_keyword(self):
        result = chat(self.USER_ID, "I have fever")
        assert "fever" in result["reply"].lower()

    def test_headache_keyword(self):
        result = chat(self.USER_ID, "I have a bad headache")
        assert "headache" in result["reply"].lower()

    def test_generic_reply(self):
        result = chat(self.USER_ID, "hello")
        assert len(result["reply"]) > 10  # some response returned

    def test_disclaimer_present(self):
        result = chat(self.USER_ID, "I have chest pain")
        assert "medical advice" in result["reply"].lower() or "doctor" in result["reply"].lower()

    def test_english_detection(self):
        result = chat(self.USER_ID, "I have fever and cough")
        assert result["detected_lang"] == "english"

    def test_translated_is_none_for_english(self):
        result = chat(self.USER_ID, "I have fever")
        assert result["translated"] is None


class TestChatMemory:
    """Test conversation history management."""

    USER_ID = "test-user-002"

    def setup_method(self):
        reset_memory(self.USER_ID)

    def test_history_empty_after_reset(self):
        history = get_history(self.USER_ID)
        assert history == []

    def test_history_grows(self):
        chat(self.USER_ID, "hello")
        history = get_history(self.USER_ID)
        assert len(history) == 2  # 1 human + 1 ai

    def test_multi_turn(self):
        chat(self.USER_ID, "I have fever")
        chat(self.USER_ID, "How long should I rest?")
        history = get_history(self.USER_ID)
        assert len(history) == 4  # 2 human + 2 ai

    def test_reset_clears(self):
        chat(self.USER_ID, "test")
        reset_memory(self.USER_ID)
        assert get_history(self.USER_ID) == []
