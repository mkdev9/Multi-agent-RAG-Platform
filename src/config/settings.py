"""Application configuration settings."""
import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    url: str = Field(
        default="postgresql+asyncpg://rag_user:password@localhost:5432/rag_db",
        env="DATABASE_URL"
    )
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""
    
    provider: str = Field(default="chroma", env="VECTOR_DB_PROVIDER")
    chroma_path: str = Field(default="./data/chroma", env="CHROMA_PATH")
    chroma_host: Optional[str] = Field(default=None, env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    collection_name: str = Field(default="knowledge_base", env="VECTOR_COLLECTION_NAME")


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    
    default_provider: str = Field(default="openai", env="LLM_DEFAULT_PROVIDER")
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    
    # Azure OpenAI
    azure_openai_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_version: str = Field(default="2023-12-01-preview", env="AZURE_OPENAI_VERSION")
    
    # AWS Bedrock
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")


class AuthSettings(BaseSettings):
    """Authentication configuration."""
    
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="AUTH_SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="AUTH_ALGORITHM")
    access_token_expire_minutes: int = Field(default=15, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=30, env="REFRESH_TOKEN_EXPIRE_DAYS")


class StorageSettings(BaseSettings):
    """File storage configuration."""
    
    provider: str = Field(default="local", env="STORAGE_PROVIDER")
    local_path: str = Field(default="./data/files", env="STORAGE_LOCAL_PATH")
    
    # AWS S3
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    aws_s3_region: str = Field(default="us-east-1", env="AWS_S3_REGION")
    
    # Azure Blob
    azure_storage_account: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: Optional[str] = Field(default=None, env="AZURE_STORAGE_KEY")
    azure_container: Optional[str] = Field(default=None, env="AZURE_CONTAINER")


class APISettings(BaseSettings):
    """API configuration."""
    
    title: str = "RAG Knowledge Platform"
    description: str = "RAG-based knowledge platform with multi-agent orchestration"
    version: str = "0.1.0"
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")


class Settings(BaseSettings):
    """Main application settings."""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    llm: LLMSettings = LLMSettings()
    redis: RedisSettings = RedisSettings()
    auth: AuthSettings = AuthSettings()
    storage: StorageSettings = StorageSettings()
    api: APISettings = APISettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()