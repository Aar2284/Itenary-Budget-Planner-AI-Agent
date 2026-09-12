from tools.tavily_tool import tavily_search


class TestTavilySearch:
    def test_formats_results(self, monkeypatch):
        import tools.tavily_tool as tavily_tool

        fake_client = type(
            "FakeClient",
            (),
            {
                "search": staticmethod(
                    lambda query, max_results: {
                        "results": [
                            {
                                "title": "Best Hotels in Tokyo",
                                "url": "https://example.com/hotels",
                                "content": "Great hotels for every budget. " * 20,
                            },
                            {
                                "title": "Budget Stays",
                                "url": "https://example.com/budget",
                                "content": "Affordable options near stations.",
                            },
                        ]
                    }
                )
            },
        )()

        monkeypatch.setattr(tavily_tool, "client", fake_client)

        result = tavily_search("hotels in Tokyo")

        assert "1. **Best Hotels in Tokyo**" in result
        assert "https://example.com/hotels" in result
        assert "2. **Budget Stays**" in result
        assert "Affordable options near stations." in result

    def test_truncates_long_snippets(self, monkeypatch):
        import tools.tavily_tool as tavily_tool

        long_content = "word " * 200

        fake_client = type(
            "FakeClient",
            (),
            {
                "search": staticmethod(
                    lambda query, max_results: {
                        "results": [
                            {
                                "title": "Long",
                                "url": "https://example.com",
                                "content": long_content,
                            }
                        ]
                    }
                )
            },
        )()

        monkeypatch.setattr(tavily_tool, "client", fake_client)

        result = tavily_search("long content")

        assert "..." in result
        assert len(result) < len(long_content)
