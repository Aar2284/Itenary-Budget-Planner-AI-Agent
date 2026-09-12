from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

import backend


class TestTruncate:
    def test_short_text_unchanged(self):
        assert backend.truncate("hello", 100) == "hello"

    def test_none_becomes_empty(self):
        assert backend.truncate(None) == ""

    def test_long_text_truncated(self):
        text = "word " * 500
        result = backend.truncate(text, 100)
        # may exceed max_chars by up to len("...")
        assert len(result) <= 103
        assert result.endswith("...")

    def test_truncates_on_word_boundary(self):
        text = "alpha beta gamma delta"
        result = backend.truncate(text, 10)
        assert " " not in result.rstrip(".")
        assert result.endswith("...")


class TestTravelGraph:
    def test_graph_compiled(self):
        assert backend.travel_graph is not None

    def test_checkpointer_assigned(self):
        assert backend.checkpointer is not None

    def test_state_fields_present(self):
        state = backend.TravelState.__annotations__
        for field in (
            "messages",
            "user_query",
            "flight_results",
            "hotel_results",
            "itinerary",
            "llm_calls",
        ):
            assert field in state


class TestRunTravelAgent:
    def test_generates_thread_id_when_missing(self, monkeypatch):
        mock_result = {
            "messages": [AIMessage(content="final answer")],
            "flight_results": "flights",
            "hotel_results": "hotels",
            "itinerary": "day 1",
            "llm_calls": 4,
        }

        monkeypatch.setattr(
            backend.travel_graph,
            "invoke",
            lambda state, config=None: mock_result,
        )

        result = backend.run_travel_agent("Plan a trip")

        assert result["thread_id"].startswith("user_")
        assert result["answer"] == "final answer"
        assert result["llm_calls"] == 4

    def test_uses_provided_thread_id(self, monkeypatch):
        mock_result = {
            "messages": [AIMessage(content="ok")],
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        }

        captured = {}

        def fake_invoke(state, config=None):
            captured["config"] = config
            return mock_result

        monkeypatch.setattr(backend.travel_graph, "invoke", fake_invoke)

        backend.run_travel_agent("Plan a trip", thread_id="abc123")

        assert captured["config"]["configurable"]["thread_id"] == "abc123"


class TestInvokeLlm:
    def test_returns_on_success(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="ok")
        monkeypatch.setattr(backend, "llm", mock_llm)

        result = backend.invoke_llm(["msg"])
        assert result.content == "ok"
        mock_llm.invoke.assert_called_once()

    def test_retries_on_rate_limit(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            Exception("429 rate_limit exceeded"),
            AIMessage(content="recovered"),
        ]
        monkeypatch.setattr(backend, "llm", mock_llm)
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)

        result = backend.invoke_llm(["msg"])
        assert result.content == "recovered"
        assert mock_llm.invoke.call_count == 2

    def test_raises_after_max_retries(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("429 rate_limit")
        monkeypatch.setattr(backend, "llm", mock_llm)
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)

        try:
            backend.invoke_llm(["msg"], retries=2)
            raise AssertionError("expected exception")
        except Exception as e:
            assert "429" in str(e)
        assert mock_llm.invoke.call_count == 2
