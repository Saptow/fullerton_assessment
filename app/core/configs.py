from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    openai_model: str = "gpt-5.4"
    openai_image_detail: str = "high"


settings = AppConfig()
