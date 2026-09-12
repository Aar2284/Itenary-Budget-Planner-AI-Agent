from tools.flight_tool import (
    clean_text,
    country_name_to_code,
    find_location_mentions,
    format_flight,
    parse_route,
    resolve_location_to_iata,
    search_flights,
)


class TestCleanText:
    def test_lowercases_and_strips(self):
        # "trip" is a stop word and is removed
        assert clean_text("  Japan Trip  ") == "japan"

    def test_removes_punctuation(self):
        assert clean_text("Tokyo, Japan!") == "tokyo japan"

    def test_removes_stop_words(self):
        assert clean_text("flights to Paris for trip") == "to paris for"


class TestCountryNameToCode:
    def test_alias_match(self):
        assert country_name_to_code("bangladesh") == "BD"

    def test_full_country_name(self):
        assert country_name_to_code("japan") == "JP"

    def test_unknown_returns_none(self):
        assert country_name_to_code("atlantis") is None


class TestResolveLocationToIata:
    def test_direct_iata_code(self):
        assert resolve_location_to_iata("DAC") == "DAC"

    def test_country_to_main_airport(self):
        assert resolve_location_to_iata("Bangladesh") == "DAC"

    def test_city_to_main_airport(self):
        assert resolve_location_to_iata("Tokyo") == "NRT"

    def test_empty_returns_none(self):
        assert resolve_location_to_iata("") is None

    def test_unknown_returns_none(self):
        assert resolve_location_to_iata("zzzqqqxxx") is None


class TestFindLocationMentions:
    def test_finds_multiple_locations(self):
        mentions = find_location_mentions("flights from Bangladesh to Japan")
        assert "bangladesh" in mentions
        assert "japan" in mentions

    def test_no_locations(self):
        assert find_location_mentions("tell me a joke") == []


class TestParseRoute:
    def test_from_x_to_y(self):
        dep, arr = parse_route("flights from Bangladesh to Japan")
        assert dep == "DAC"
        assert arr == "NRT"

    def test_direct_iata_codes(self):
        dep, arr = parse_route("DAC to NRT")
        assert dep == "DAC"
        assert arr == "NRT"

    def test_from_only(self):
        dep, arr = parse_route("flights from Dhaka")
        assert dep == "DAC"
        assert arr is None

    def test_to_only(self):
        # "to X" branch returns destination only; default origin applies
        # only via the single-mention fallback path
        dep, arr = parse_route("flights to Tokyo")
        assert dep is None
        assert arr == "NRT"

    def test_single_mention_fallback_uses_default_origin(self):
        dep, arr = parse_route("Tokyo vacation please")
        assert dep == "DAC"
        assert arr == "NRT"

    def test_global_keywords(self):
        dep, arr = parse_route("show all flights worldwide")
        assert dep is None
        assert arr is None


class TestFormatFlight:
    def test_formats_fields(self):
        flight = {
            "airline": {"name": "Biman Bangladesh"},
            "flight": {"iata": "BG123"},
            "flight_status": "scheduled",
            "departure": {
                "airport": "Hazrat Shahjalal",
                "iata": "DAC",
                "terminal": "1",
                "gate": "A5",
                "scheduled": "2026-11-20 10:00",
                "delay": None,
            },
            "arrival": {
                "airport": "Narita",
                "iata": "NRT",
                "terminal": "2",
                "gate": "B1",
                "scheduled": "2026-11-20 16:00",
                "delay": 15,
            },
        }

        text = format_flight(flight)

        assert "Biman Bangladesh" in text
        assert "BG123" in text
        assert "scheduled" in text
        assert "DAC" in text
        assert "NRT" in text
        assert "15 minutes" in text


class TestSearchFlights:
    def test_missing_api_key(self, monkeypatch):
        import tools.flight_tool as flight_tool

        monkeypatch.setattr(flight_tool, "API_KEY", None)
        result = search_flights("flights to Tokyo")
        assert "AVIATIONSTACK_API_KEY is missing" in result

    def test_successful_response(self, monkeypatch):
        import tools.flight_tool as flight_tool

        monkeypatch.setattr(flight_tool, "API_KEY", "test-key")

        payload = {
            "data": [
                {
                    "airline": {"name": "Test Air"},
                    "flight": {"iata": "TA1"},
                    "flight_status": "scheduled",
                    "departure": {
                        "airport": "DAC",
                        "iata": "DAC",
                        "terminal": "1",
                        "gate": "A1",
                        "scheduled": "2026-11-20 10:00",
                        "delay": None,
                    },
                    "arrival": {
                        "airport": "NRT",
                        "iata": "NRT",
                        "terminal": "2",
                        "gate": "B2",
                        "scheduled": "2026-11-20 16:00",
                        "delay": None,
                    },
                }
            ]
        }

        class FakeResponse:
            def json(self):
                return payload

        monkeypatch.setattr(
            flight_tool.requests, "get", lambda *a, **k: FakeResponse()
        )

        result = search_flights("flights from Bangladesh to Japan")
        assert "Live flights from DAC to NRT" in result
        assert "Test Air" in result

    def test_api_error_payload(self, monkeypatch):
        import tools.flight_tool as flight_tool

        monkeypatch.setattr(flight_tool, "API_KEY", "test-key")

        payload = {
            "error": {"code": 101, "message": "Invalid API access key"}
        }

        class FakeResponse:
            def json(self):
                return payload

        monkeypatch.setattr(
            flight_tool.requests, "get", lambda *a, **k: FakeResponse()
        )

        result = search_flights("flights to Tokyo")
        assert "Flight API error" in result
        assert "Invalid API access key" in result

    def test_empty_results(self, monkeypatch):
        import tools.flight_tool as flight_tool

        monkeypatch.setattr(flight_tool, "API_KEY", "test-key")

        class FakeResponse:
            def json(self):
                return {"data": []}

        monkeypatch.setattr(
            flight_tool.requests, "get", lambda *a, **k: FakeResponse()
        )

        result = search_flights("flights from DAC to NRT")
        assert "No live flight data found" in result
        assert "DAC to NRT" in result
