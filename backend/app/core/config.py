from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://impactai:impactai@localhost:5432/impactai"
    data_dir: Path = Path("data/source")
    artifact_dir: Path = Path("ml/artifacts")
    peer_interval_dir: Path = Path("data/research/peer-intervals-20261017")
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    coach_llm_url: str = ""
    coach_llm_api_key: str = ""
    coach_llm_model: str = ""


settings = Settings()
