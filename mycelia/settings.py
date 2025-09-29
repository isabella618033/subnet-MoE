from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    role: str = "miner"  # or validator
    miner_id: str = "miner-001"
    model_name: str = "meta-llama/Llama-3-8B"
    device: str = "cuda"
    seed: int = 42
    broker_url: str = "redis://redis:6379/0"
    s3_bucket: str = "mycelia"
    private_key: str = "CHANGE_ME"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
