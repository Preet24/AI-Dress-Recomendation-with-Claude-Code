"""Application settings, loaded from environment variables and `.env`."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings; each field can be overridden with an `OUTFIT_<FIELD>` env var."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OUTFIT_", extra="ignore")

    # Segmentation (rembg ONNX models, always on CPU)
    # garment alone: flat-lay, hanger, product shot. birefnet-general won a bake-off on 7
    # real bed photos (the only one that cut the bed out between trouser legs); ~26 s a
    # photo on CPU vs 2 s for isnet-general-use, which lost half a striped shirt.
    flat_model: str = "birefnet-general"
    worn_model: str = "u2net_cloth_seg"  # garments worn by a person
    max_side: int = Field(default=1024, ge=256)  # downscale before inference
    min_area_ratio: float = Field(default=0.02, gt=0, lt=1)  # smaller regions are dropped

    # Image embeddings (open_clip, always on CPU)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    embed_batch_size: int = Field(default=16, ge=1)

    # Vision tagging (DeepSeek API, OpenAI-compatible: the only step that sends data off
    # the machine)
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    tagger_base_url: str = "https://api.deepseek.com"
    tagger_model: str = "deepseek-flash"
    tagger_max_tokens: int = Field(default=4096, ge=256)

    # Agents (Phase 4): same DeepSeek client, text only
    agent_model: str = "deepseek-flash"
    agent_max_tokens: int = Field(default=2048, ge=256)
    # The rubric reasons longest: in the 3x eval (79 calls) median 321 reasoning tokens,
    # but 2 calls spent all 2048 thinking and returned nothing; others peaked at 896.
    critic_max_tokens: int = Field(default=4096, ge=256)

    # Storage
    raw_dir: Path = Path("data/wardrobe_raw")  # photos uploaded through the API
    clean_dir: Path = Path("data/wardrobe_clean")
    qdrant_path: Path = Path("qdrant_data")
    conversations_dir: Path = Path("data/conversations")  # chat.py --record
    profile_path: Path = Path("data/profile.json")  # the user's profile (one per install)
    candidates_dir: Path = Path("data/candidates")  # shopping: items checked, not yet bought
    wear_log_path: Path = Path("data/wear_log.json")  # outfits worn, with their ratings

    # API (Phase 6)
    max_upload_mb: float = Field(default=15, gt=0)  # phone photos are ~2-8 MB
    record_chats: bool = False  # like chat.py --record: each turn to conversations_dir
