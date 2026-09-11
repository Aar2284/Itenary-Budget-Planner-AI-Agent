import os

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("AVIATIONSTACK_API_KEY", "test-aviation-key")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/test_db",
)
os.environ.setdefault("DEFAULT_ORIGIN_IATA", "DAC")
