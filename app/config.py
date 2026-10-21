"""Typed application configuration, loaded from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Infrastructure ----------------------------------------------------
    database_url: str = "postgresql+asyncpg://alertradar:alertradar@db:5432/alertradar"
    redis_url: str = "redis://redis:6379/0"
    db_pool_size: int = 10
    db_max_overflow: int = 10

    # --- Gemini ------------------------------------------------------------
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash"
    #: Empty string means "omit thinking_config" — required for models that use
    #: thinking_budget instead of thinking_level (the Gemini 2.5 family).
    llm_thinking_level: str = "low"
    #: How aggressively Gemini's content filters block. See .env.example.
    llm_safety_threshold: str = "BLOCK_ONLY_HIGH"
    llm_max_tokens: int = 8192
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    llm_concurrency: int = 2
    #: Hard requests-per-day cap to ration against (free tier: 500 on
    #: flash-lite, 20 on flash). 0 disables rationing entirely.
    llm_daily_request_budget: int = 500
    #: Slice of the day's budget spendable ahead of schedule, so a barrage is
    #: not throttled just because it started early. See RequestBudget.
    llm_budget_burst_ratio: float = 0.15
    #: Timezone the provider resets the daily quota in. Google's free tier
    #: rolls over at Pacific midnight — which is 10:00 in Kyiv, i.e. right
    #: after the overnight attack window rather than in the middle of it.
    llm_budget_timezone: str = "America/Los_Angeles"

    # --- Telegram ----------------------------------------------------------
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_session: str = ""
    #: `NoDecode` is load-bearing: without it pydantic-settings tries to
    #: JSON-decode any list-typed field straight from the env var, and dies on
    #: a plain comma-separated string before the validator below ever runs.
    telegram_channels: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- Batching ----------------------------------------------------------
    batch_window_seconds: float = 45.0
    batch_max_messages: int = 60
    batch_min_message_chars: int = 8
    batch_queue_maxsize: int = 2000
    #: Verbatim reposts seen this recently are dropped before the model sees
    #: them. Must stay well under `default_ttl_minutes`; see RepostFilter.
    #: 0 disables cross-window suppression entirely.
    batch_repost_ttl_seconds: float = 300.0
    #: How long a confirmed threat keeps the prefilter open, letting the
    #: channels' running commentary through. 0 disables. See AttackMode.
    attack_mode_ttl_seconds: float = 900.0

    # --- Threat model ------------------------------------------------------
    default_ttl_minutes: int = 20
    min_confidence: float = 0.35
    high_confidence: float = 0.65
    #: The safety margin is a *fraction of the target's own size*, not a fixed
    #: distance. The gazetteer spans three orders of magnitude — a Kyiv
    #: district is ~5 km across, an oblast ~150 km — so any single constant
    #: either blankets a city or vanishes around an oblast. 15 km on
    #: Дарницький (11 km across) put all of Kyiv under one warning.
    #:
    #: Applied against sqrt(core area), which is the sector's characteristic
    #: width in metres, so 0.25 reads as "a quarter of the target's width".
    buffer_ratio_shahed: float = 0.25
    buffer_ratio_missile: float = 0.30
    buffer_ratio_recon: float = 0.18
    #: Clamp on the result. The floor keeps a 3 km² district from getting a
    #: margin thinner than the positional error of the reports themselves; the
    #: ceiling stops an oblast-wide warning from swallowing its neighbours.
    buffer_floor_meters: int = 2_000
    buffer_ceiling_meters: int = 30_000
    point_radius_meters: int = 12_000
    sector_match_threshold: float = 0.45

    # --- ETA estimation ----------------------------------------------------
    #: Nominal cruise speeds, km/h. Only used when the source did not state an
    #: arrival time itself — which, on real traffic, is nearly always. These are
    #: published figures, not measurements: calibrate them against your own
    #: history before trusting the tighter end of the window.
    cruise_kmh_shahed: int = 180
    cruise_kmh_missile: int = 800
    cruise_kmh_recon: int = 140
    #: Below this separation, "approaching from" carries no information — the
    #: origin sector already touches the target — and the estimate is dropped
    #: rather than reported as "now".
    eta_min_approach_meters: int = 5_000

    # --- Hot path ----------------------------------------------------------
    snapshot_poll_seconds: float = 2.0
    #: Douglas-Peucker tolerance for the map overlay, in degrees (~0.005 is
    #: 400-550 m at Ukraine's latitude). Display only — the geometry used for
    #: point-in-polygon lookups is never simplified.
    zones_simplify_degrees: float = 0.005
    #: Decimal places kept in overlay coordinates. 4 is ~11 m.
    zones_precision: int = 4
    #: Hard ceiling on the vertices of any single overlay shape. A tolerance
    #: bounds a shape's error, not its size: at 0.005 degrees one Kyiv district
    #: thins to 21 vertices but a whole-oblast union still lands at ~690, so
    #: the payload grows with how much of the country is under alert. The
    #: tolerance escalates until a shape fits this budget. 0 disables the cap.
    zones_max_vertices: int = 200
    janitor_interval_seconds: float = 60.0

    log_level: str = "INFO"

    @field_validator("telegram_channels", mode="before")
    @classmethod
    def _split_channels(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    def cruise_kmh_for(self, threat_type: str) -> int:
        return {
            "shahed": self.cruise_kmh_shahed,
            "missile": self.cruise_kmh_missile,
            "recon": self.cruise_kmh_recon,
        }.get(threat_type, self.cruise_kmh_shahed)

    def buffer_ratio_for(self, threat_type: str) -> float:
        return {
            "shahed": self.buffer_ratio_shahed,
            "missile": self.buffer_ratio_missile,
            "recon": self.buffer_ratio_recon,
        }.get(threat_type, self.buffer_ratio_shahed)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
