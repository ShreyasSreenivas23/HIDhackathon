from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = Field(default=None, alias="AWS_SESSION_TOKEN")

    challenge_signing_key: str = Field(alias="CHALLENGE_SIGNING_KEY")
    challenge_ttl_seconds: int = Field(default=15, alias="CHALLENGE_TTL_SECONDS")
    enrolled_faces_bucket: str = Field(default="hid-goaldigger-enrolled-faces", alias="ENROLLED_FACES_BUCKET")
    rekognition_collection_id: str = Field(default="hid-hackathon-collection", alias="REKOGNITION_COLLECTION_ID")
    ddb_table: str = Field(default="AccessAuditLogs", alias="DDB_TABLE")
    bedrock_model_id: str = Field(default="amazon.nova-lite-v1:0", alias="BEDROCK_MODEL_ID")
    fast_path_timeout_ms: int = Field(default=1200, alias="FAST_PATH_TIMEOUT_MS")
    step_up_timeout_ms: int = Field(default=800, alias="STEP_UP_TIMEOUT_MS")


settings = Settings()