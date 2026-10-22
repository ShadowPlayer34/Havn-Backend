"""Telegram ingestion + the central AI agent loop.

Pipeline
--------
    Telethon events
        -> cheap keyword prefilter        (drops most traffic before it costs money)
        -> in-window near-duplicate drop  (channels relay each other verbatim)
        -> cross-window repost drop       (a window of pure reposts costs no call)
        -> asyncio.Queue
        -> 30-60s batching window         (one LLM call per window, not per message)
        -> structured extraction
        -> sector resolution -> PostGIS geometry
        -> deduplicating upsert
        -> Redis snapshot publish

A separate janitor archives expired clusters and republishes the snapshot so
the API sees expiry even during quiet periods.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import re
import signal
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from telethon import TelegramClient, events
from telethon.sessions import StringSession

from app import cache, database, geo
from app.bootstrap import ensure_schema
from app.config import Settings, get_settings
from app.database import ThreatUpsert
from app.llm import (
    BatchMessage,
    LLMExtractionError,
    LLMRefusal,
    LLMRequestError,
    LLMTransient,
    ThreatExtractor,
)
from app.schemas import ExtractedThreat, normalize_sector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------

#: Cheapest possible cost control: most messages in a monitoring channel are
#: not threat reports. Anything that clears this regex goes to the model;
#: anything that does not is free. Tune it against your channels — a false
#: negative here is a missed threat, so keep it generous.
#:
#: Measured against 6h of live traffic from the configured channels, so the
#: shorthand is not guesswork: "КР" (крилаті ракети) and "реактив" (jet-powered
#: Geran) are the two most common threat words on the wire, and neither is
#: spelled out. Likewise `курс` appears far more often bare — "Курс Кролевець",
#: "курс сектор Пирятин" — than in the "курсом на" form.
_RELEVANCE_PATTERN = re.compile(
    # Munitions, spelled out.
    r"шахед|шахід|герань|geran|мопед|бпла|бплa|дрон|drone|"
    r"ракет|missile|калібр|kalibr|кинджал|kinzhal|іскандер|искандер|iskander|"
    r"х-101|х-555|х-59|kh-101|онікс|оникс|onyx|с-300|с-400|"
    # Munitions, as the channels actually abbreviate them.
    r"\bкр\b|\bбр\b|\bкаб\b|реактив|швидкісн|мандрівник|"
    # Reconnaissance.
    r"орлан|zala|supercam|розвід|развед|recon|"
    # Alerts and all-clears.
    r"тривог|тривож|загроз|повітрян|відбій|отбой|увага|"
    # Movement, grouping and the imperative to react. These carry most of the
    # running commentary during a wave.
    r"\bгруп|\bціл[ьіяе]|\bсектор|\bкурс|\bрух|заліт|зайш|ланцюг|"
    r"підліт|наближ|розворот|зниження|зреагу|фіксац|"
    r"у\s+напрямку|в\s+напрямку|злет|взлет|пуск|запуск|"
    # Carriers that have not launched yet.
    r"ту-95|ту-160|ту-22|міг-31|миг-31|mig-31",
    re.IGNORECASE | re.UNICODE,
)


class AttackMode:
    """Opens the prefilter for as long as a wave is already in progress.

    The keyword filter judges each message alone, but these channels post a
    running commentary: once something is airborne, "Рожівка на Рожни далі",
    "Вздовж Київ правий" and "Розворот на Бровари!" are position updates that
    contain no munition vocabulary at all. No word list can catch them, because
    the subject lives in the preceding messages rather than in this one — on
    live traffic they are a fifth of everything the channels post.

    So the regex is left to do only what it can: decide whether anything is
    happening at all. Once an extraction confirms that something is, this opens
    and every message passes until the channels have been quiet for `ttl`.

    Cheaper than it looks. Windows close on a timer, not on a message count, so
    admitting more messages makes each window *fuller* rather than making more
    of them — the LLM call count is unchanged and only the prompt grows.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._until = 0.0

    @property
    def is_open(self) -> bool:
        if self._ttl <= 0:
            return False
        return asyncio.get_running_loop().time() < self._until

    def extend(self) -> None:
        """Called whenever a window yields a real threat."""
        was_open = self.is_open
        self._until = asyncio.get_running_loop().time() + self._ttl
        if not was_open:
            logger.info("attack_mode_opened ttl=%.0fs", self._ttl)


def is_relevant(text: str, min_chars: int, attack_open: bool = False) -> bool:
    if len(text.strip()) < min_chars:
        return False
    if attack_open:
        # Mid-wave every message from these channels is about the wave.
        return True
    return _RELEVANCE_PATTERN.search(text) is not None


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IncomingMessage:
    channel: str
    message_id: int
    timestamp: datetime
    text: str


class Batcher:
    """Accumulates messages into fixed time windows.

    A window closes on whichever comes first: ``batch_window_seconds`` since the
    *first* message in it, or ``batch_max_messages``. That bounds both latency
    (a threat is never held longer than one window) and prompt size (a barrage
    cannot produce a 500-message prompt).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(
            maxsize=settings.batch_queue_maxsize
        )
        self._reposts = RepostFilter(settings.batch_repost_ttl_seconds)
        self.dropped = 0
        self.suppressed = 0

    def offer(self, message: IncomingMessage) -> None:
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            # Shedding load is correct here: an overflowing queue means we are
            # already behind, and stale threats have no value.
            self.dropped += 1
            logger.warning("ingest_queue_full dropped_total=%d", self.dropped)

    async def next_window(self) -> list[IncomingMessage]:
        first = await self._queue.get()
        window: list[IncomingMessage] = [first]
        deadline = asyncio.get_running_loop().time() + self._settings.batch_window_seconds

        while len(window) < self._settings.batch_max_messages:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            window.append(item)

        deduped = _dedupe_window(window)
        fresh = self._reposts.filter(deduped)
        if deduped and not fresh:
            self.suppressed += 1
            logger.info(
                "window_all_reposts messages=%d suppressed_total=%d",
                len(deduped),
                self.suppressed,
            )
        return fresh


def _dedupe_window(messages: list[IncomingMessage]) -> list[IncomingMessage]:
    """Drop verbatim reposts within a window — pure token savings."""
    seen: set[str] = set()
    unique: list[IncomingMessage] = []
    for message in messages:
        digest = _content_hash(message.text)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(message)
    return unique


class RepostFilter:
    """Drops verbatim reposts that straddle windows — pure *call* savings.

    `_dedupe_window` catches the copies that land inside one window. This
    catches the ones that do not, which is where the call volume actually is: a
    monitoring channel repeats the same sighting once a minute for as long as
    the group is airborne, so one 20-minute drone transit can otherwise be
    twenty-odd windows all describing it. When suppression empties a window
    completely, no LLM call happens at all.

    Suppressing is safe because re-extracting identical text yields an
    identical `dedup_key`, and `upsert_threat` merges with GREATEST() rather
    than overwriting — the numbers cannot go backwards. The one thing a
    suppressed repost does not do is push `expires_at` forward, so the TTL here
    must stay comfortably below `default_ttl_minutes`: a threat still being
    reposted gets a real call, and a fresh countdown, every `ttl_seconds`.

    Matching is on exact normalised text, so a correction — different count,
    different direction, an all-clear — hashes differently and always passes.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    def filter(self, messages: list[IncomingMessage]) -> list[IncomingMessage]:
        if self._ttl <= 0:
            return messages
        now = asyncio.get_running_loop().time()
        self._prune(now)
        fresh: list[IncomingMessage] = []
        for message in messages:
            digest = _content_hash(message.text)
            if digest in self._seen:
                continue
            self._seen[digest] = now
            fresh.append(message)
        return fresh

    def _prune(self, now: float) -> None:
        """Expire old hashes so the map stays bounded by traffic, not uptime."""
        stale = [d for d, seen_at in self._seen.items() if now - seen_at >= self._ttl]
        for digest in stale:
            del self._seen[digest]


# ---------------------------------------------------------------------------
# Request rationing
# ---------------------------------------------------------------------------


class RequestBudget:
    """Rations a hard requests-per-day quota so the system degrades, not dies.

    The free tier is a daily bucket that resets at Pacific midnight — 500
    requests on flash-lite, 20 on flash. Requests-per-minute never binds here:
    one window every 45s with `llm_concurrency` of 2 peaks around 1.3 RPM
    against a limit of 15. The day's total is the only real constraint.

    Spending it greedily is the failure mode worth engineering against. Burn
    the last request at 02:00 and the map goes blind for the rest of the night,
    which is exactly the stretch it exists for — and it happens silently, as a
    stream of 429s the pipeline treats as ordinary transient errors.

    So the budget accrues with the clock. A window may spend if the day's usage
    is still under `elapsed_fraction + burst_ratio` of the cap; the burst slice
    is what lets a barrage front-load rather than being throttled for starting
    early. Denied windows are dropped and logged as such — no call, no 429.

    The counter lives in Redis, so restarting the worker does not hand it a
    fresh budget it has already spent.
    """

    def __init__(self, settings: Settings, redis: Redis) -> None:
        self._settings = settings
        self._redis = redis
        self._tz = ZoneInfo(settings.llm_budget_timezone)
        self.denied = 0

    def _day_bounds(self, now: datetime) -> tuple[datetime, datetime]:
        """Start and end of the quota day *containing* `now`."""
        local = now.astimezone(self._tz)
        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)

    async def claim(self) -> bool:
        """Reserve one request. False means "do not call the model"."""
        cap = self._settings.llm_daily_request_budget
        if cap <= 0:
            return True

        now = datetime.now(UTC)
        start, end = self._day_bounds(now)
        key = f"alertradar:llm:spend:{start.date().isoformat()}"

        # Claim first, hand back on refusal. Reading before incrementing would
        # let two concurrent windows both see the last slot as free.
        used = int(await self._redis.incr(key))
        if used == 1:
            # Outlive the quota day by an hour so a clock skew cannot resurrect
            # yesterday's counter as today's.
            await self._redis.expire(key, int((end - now).total_seconds()) + 3600)

        elapsed = (now - start).total_seconds() / (end - start).total_seconds()
        allowance = cap * min(1.0, elapsed + self._settings.llm_budget_burst_ratio)

        if used > allowance:
            await self._redis.decr(key)
            self.denied += 1
            logger.warning(
                "llm_budget_throttled used=%d/%d allowance=%.0f denied_total=%d",
                used - 1,
                cap,
                allowance,
                self.denied,
            )
            return False

        logger.info("llm_budget_claimed used=%d/%d allowance=%.0f", used, cap, allowance)
        return True


# ---------------------------------------------------------------------------
# Window processing
# ---------------------------------------------------------------------------


class WindowProcessor:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        redis: Redis,
        extractor: ThreatExtractor,
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._redis = redis
        self._extractor = extractor
        self._budget = RequestBudget(settings, redis)
        self.attack = AttackMode(settings.attack_mode_ttl_seconds)
        self._semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def process(self, window: list[IncomingMessage]) -> None:
        async with self._semaphore:
            # Inside the semaphore: the claim should reflect calls actually
            # about to happen, not calls queued behind a slow one.
            if not await self._budget.claim():
                return
            try:
                await self._process(window)
            except LLMRefusal as exc:
                # Not retryable with the same payload; log loudly and move on
                # rather than stalling ingestion on one window. A rising rate
                # here means the safety threshold needs revisiting.
                logger.error("llm_refused messages=%d detail=%s", len(window), exc)
            except LLMExtractionError as exc:
                logger.error("llm_bad_output messages=%d detail=%s", len(window), exc)
            except LLMRequestError as exc:
                # Misconfiguration — wrong model id, bad key, rejected schema.
                # Every window will fail the same way until a human fixes it.
                logger.error("llm_request_rejected detail=%s", exc)
            except LLMTransient as exc:
                # The SDK already retried; a window is cheap to lose.
                logger.warning("llm_transient_failure detail=%s", exc)
            except Exception:
                logger.exception("window_processing_failed messages=%d", len(window))

    async def _process(self, window: list[IncomingMessage]) -> None:
        batch = [
            BatchMessage(
                channel=m.channel,
                message_id=m.message_id,
                timestamp=m.timestamp,
                text=m.text,
            )
            for m in window
        ]
        result = await self._extractor.extract(batch)

        if not result.threats and not result.all_clear_sectors:
            logger.info("window_empty messages=%d", len(window))
            return

        stored = 0
        async with database.session_scope(self._sessionmaker) as session:
            for threat in result.threats:
                if threat.confidence_score < self._settings.min_confidence:
                    logger.info(
                        "threat_below_confidence type=%s score=%.2f",
                        threat.threat_type,
                        threat.confidence_score,
                    )
                    continue
                if await self._persist(session, threat, window):
                    stored += 1

            if result.all_clear_sectors:
                normalized = [normalize_sector(s) for s in result.all_clear_sectors]
                cleared = await database.expire_sectors(session, normalized)
                logger.info(
                    "all_clear sectors=%s expired=%d", result.all_clear_sectors, cleared
                )

        if stored:
            # Something is genuinely in the air: let the commentary through
            # until the channels go quiet again.
            self.attack.extend()

        logger.info(
            "window_processed messages=%d extracted=%d stored=%d attack_mode=%s",
            len(window),
            len(result.threats),
            stored,
            self.attack.is_open,
        )
        await self.publish_snapshot()

    async def _persist(
        self,
        session: AsyncSession,
        threat: ExtractedThreat,
        window: list[IncomingMessage],
    ) -> bool:
        resolved = await geo.resolve_sectors(
            session,
            threat.target_sectors,
            self._settings.sector_match_threshold,
            regions=threat.regions_by_sector(),
            fuzzy_max_meters=self._settings.sector_fuzzy_max_km * 1000.0,
            region_threshold=self._settings.sector_region_match_threshold,
        )
        if not resolved.is_localisable:
            # No geometry means no spatial answer — storing it would only
            # inflate the index. Surfaced as a log line so gazetteer gaps are
            # visible and fixable.
            logger.warning(
                "threat_unlocalised type=%s sectors=%s unmatched=%s",
                threat.threat_type,
                threat.target_sectors,
                resolved.unmatched,
            )
            return False

        origin = await geo.resolve_origin(
            session, threat.origin_sector, self._settings.sector_match_threshold
        )
        if threat.origin_sector and origin is None:
            # Not fatal — the threat is still localised by its targets, it just
            # gets a plain radius instead of a corridor. Logged because it is
            # the same gazetteer gap `threat_unlocalised` reports, seen from
            # the other side, and it silently costs the client its ORIGIN row.
            logger.info(
                "origin_unmatched type=%s origin=%r",
                threat.threat_type,
                threat.origin_sector,
            )
        geometry = await geo.build_risk_geometry(
            session,
            resolved,
            buffer_ratio=self._settings.buffer_ratio_for(threat.threat_type),
            buffer_floor_meters=self._settings.buffer_floor_meters,
            buffer_ceiling_meters=self._settings.buffer_ceiling_meters,
            point_radius_meters=self._settings.point_radius_meters,
            origin_sector_id=origin.sector_id if origin else None,
        )
        if geometry is None:
            logger.warning("threat_geometry_failed type=%s", threat.threat_type)
            return False

        # A stated arrival time always wins — the channel can see things a
        # speed constant cannot. The estimate only fills the (very common) gap
        # where no message mentioned one.
        eta_minutes = threat.eta_minutes
        eta_source = "reported" if eta_minutes is not None else None
        if eta_minutes is None:
            eta_minutes = geo.estimate_eta_minutes(
                geometry.approach_meters,
                self._settings.cruise_kmh_for(threat.threat_type),
                self._settings.eta_min_approach_meters,
            )
            if eta_minutes is not None:
                eta_source = "estimated"
                logger.info(
                    "eta_estimated type=%s approach_km=%.0f eta=%dmin",
                    threat.threat_type,
                    (geometry.approach_meters or 0) / 1000.0,
                    eta_minutes,
                )

        cited = _cited_messages(threat.source_refs, window)

        await database.upsert_threat(
            session,
            ThreatUpsert(
                dedup_key=threat.dedup_key(),
                threat_type=str(threat.threat_type),
                count=threat.count,
                confidence_score=threat.confidence_score,
                sectors=resolved.normalized_names,
                ttl_minutes=threat.ttl_minutes or self._settings.default_ttl_minutes,
                eta_minutes=eta_minutes,
                origin_sector=origin.name if origin else None,
                core_ewkb=geometry.core_ewkb,
                risk_ewkb=geometry.risk_ewkb,
                source_channel=window[0].channel if window else None,
                source_message_ids=[message.message_id for message in cited],
                raw={
                    "extracted": threat.model_dump(mode="json"),
                    # The text the extraction was made from. Telegram ids alone
                    # cannot answer "why is Запоріжжя in a Kyiv recon threat"
                    # once the channel has moved on, and `raw` is replaced on
                    # every re-report, so these always belong to `extracted`.
                    "source_messages": [
                        {
                            "channel": message.channel,
                            "id": message.message_id,
                            "at": message.timestamp.isoformat(),
                            "text": message.text,
                        }
                        for message in cited
                    ],
                    "matched_sectors": resolved.matched_names,
                    "unmatched_sectors": resolved.unmatched,
                    "too_far_sectors": resolved.too_far,
                    "region_fallback_sectors": resolved.region_fallbacks,
                    "eta_source": eta_source,
                    "approach_meters": geometry.approach_meters,
                },
            ),
        )
        return True

    async def publish_snapshot(self) -> None:
        async with database.session_scope(self._sessionmaker) as session:
            records = await database.load_active_threats(
                session, self._settings.min_confidence
            )
        await cache.publish_snapshot(self._redis, records)


def _cited_messages(
    refs: list[int], window: list[IncomingMessage]
) -> list[IncomingMessage]:
    """Map model-cited batch indices back to the messages they name, once each."""
    cited: list[IncomingMessage] = []
    for ref in dict.fromkeys(refs):
        if 0 <= ref < len(window):
            cited.append(window[ref])
        else:
            logger.warning("source_ref_out_of_range ref=%d window=%d", ref, len(window))
    return cited


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


async def batching_loop(batcher: Batcher, processor: WindowProcessor) -> None:
    tasks: set[asyncio.Task[None]] = set()
    try:
        while True:
            window = await batcher.next_window()
            if not window:
                continue
            # Processing runs off the batching path so a slow LLM call never
            # delays the next window; concurrency is bounded inside process().
            task = asyncio.create_task(processor.process(window))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def janitor_loop(
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    processor: WindowProcessor,
) -> None:
    """Archive lapsed clusters and keep the published snapshot honest."""
    while True:
        await asyncio.sleep(settings.janitor_interval_seconds)
        try:
            async with database.session_scope(sessionmaker) as session:
                purged = await database.purge_expired(session)
            if purged:
                logger.info("janitor_purged threats=%d", purged)
            # Republish unconditionally: TTL expiry is a state change the API
            # must see even when no new messages arrived.
            await processor.publish_snapshot()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("janitor_failed")


async def ingestion_loop(
    settings: Settings, batcher: Batcher, attack: AttackMode
) -> None:
    if not settings.telegram_channels:
        logger.warning("no TELEGRAM_CHANNELS configured — ingestion disabled")
        await asyncio.Event().wait()
        return

    client = TelegramClient(
        StringSession(settings.telegram_session),
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )

    @client.on(events.NewMessage(chats=settings.telegram_channels))
    async def _handler(event: events.NewMessage.Event) -> None:  # pragma: no cover
        text = event.message.message or ""
        if not is_relevant(text, settings.batch_min_message_chars, attack.is_open):
            return
        chat = await event.get_chat()
        channel = getattr(chat, "username", None) or str(getattr(chat, "id", "unknown"))
        batcher.offer(
            IncomingMessage(
                channel=channel,
                message_id=event.message.id,
                timestamp=event.message.date or datetime.now(UTC),
                text=text,
            )
        )

    await client.start()
    logger.info("telegram_connected channels=%s", settings.telegram_channels)
    try:
        await client.run_until_disconnected()
    finally:
        await client.disconnect()


async def run() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    engine: AsyncEngine = database.create_engine(settings)
    sessionmaker = database.create_sessionmaker(engine)
    redis = cache.create_redis(settings.redis_url)
    extractor = ThreatExtractor(settings)

    await ensure_schema(engine)

    if not extractor.is_configured:
        logger.warning(
            "GEMINI_API_KEY is not set — extraction is disabled. Ingestion, TTL "
            "expiry and snapshot publishing still run; every window will log "
            "llm_request_rejected until a key is provided."
        )

    batcher = Batcher(settings)
    processor = WindowProcessor(settings, sessionmaker, redis, extractor)
    await processor.publish_snapshot()  # make the API's first read non-empty

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tasks = [
        asyncio.create_task(
            ingestion_loop(settings, batcher, processor.attack), name="ingestion"
        ),
        asyncio.create_task(batching_loop(batcher, processor), name="batching"),
        asyncio.create_task(janitor_loop(settings, sessionmaker, processor), name="janitor"),
    ]
    stopper = asyncio.create_task(stop.wait(), name="stop")

    done, _ = await asyncio.wait(
        [*tasks, stopper], return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
        if task is not stopper and task.exception() is not None:
            logger.error("worker_task_failed name=%s", task.get_name(), exc_info=task.exception())

    logger.info("worker_shutting_down")
    for task in [*tasks, stopper]:
        task.cancel()
    await asyncio.gather(*tasks, stopper, return_exceptions=True)

    await extractor.aclose()
    await redis.aclose()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
