from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestHomePage:
    def test_home_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestFavicon:
    def test_favicon_returns_empty_json(self):
        response = client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.json() == {}


class TestTravelApi:
    def test_empty_message_returns_400(self):
        response = client.post("/api/travel", json={"message": "   "})
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    def test_successful_travel_request(self):
        fake_result = {
            "thread_id": "user_test123",
            "answer": "Here is your trip plan",
            "flight_results": "flights data",
            "hotel_results": "hotels data",
            "itinerary": "Day 1: Arrive",
            "llm_calls": 4,
        }

        with patch("app.run_travel_agent", return_value=fake_result):
            response = client.post(
                "/api/travel",
                json={"message": "Plan a trip to Tokyo", "thread_id": "abc"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["thread_id"] == "user_test123"
        assert data["answer"] == "Here is your trip plan"
        assert data["llm_calls"] == 4

    def test_agent_failure_returns_500(self):
        with patch(
            "app.run_travel_agent",
            side_effect=RuntimeError("agent exploded"),
        ):
            response = client.post(
                "/api/travel", json={"message": "Plan a trip"}
            )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "agent exploded" in data["error"]
