# AlertRadar Backend

Telegram → LLM → PostGIS → iOS. Unstructured monitoring-channel messages are
batched into 30–60 second windows, passed through a single Gemini call per
window with a strict JSON schema, resolved to administrative-district geometry,
deduplicated into TTL'd threat clusters, and served to the app as a five-field
point query.

```
 Telegram channels
        │  Telethon
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │ worker                                                   │
 │   keyword prefilter ─► dedupe ─► 30–60s window           │
 │        └─► 1 LLM call ─► sector resolve ─► PostGIS upsert │
 │   janitor: archive expired, republish snapshot           │
 └───────────────┬───────────────────────┬──────────────────┘
                 │ source of truth       │ derived snapshot
                 ▼                       ▼
          PostgreSQL + PostGIS         Redis
                 ▲                       │ version poll (2s)
                 │ fallback              ▼
 ┌───────────────┴───────────────────────────────────────────┐
 │ api (N replicas)   in-process Shapely STRtree              │
 │   GET /api/v1/threats/check  ─►  zero network I/O          │
 └────────────────────────────────────────────────────────────┘
```

## Files

| Path | Role |
|---|---|
| `app/worker.py` | Telegram ingestion, batching windows, the agent loop, janitor |
| `app/llm.py` | Gemini client, system prompt, safety config, error translation |
| `app/schemas.py` | Pydantic v2 contracts (LLM + API) and name normalisation |
| `app/database.py` | SQLAlchemy 2.0 models, dedup upsert, TTL cleanup, spatial queries |
| `app/geo.py` | District-name resolution and PostGIS risk-geometry construction |
| `app/cache.py` | Redis snapshot publish/consume |
| `app/spatial.py` | In-process STRtree index, risk grading, ETA formatting |
| `app/main.py` | FastAPI app, background refresher |
| `app/bootstrap.py` | Extension + table bootstrap (replace with Alembic for prod) |
| `scripts/login.py` | One-time Telegram login → `TELEGRAM_SESSION` |
| `scripts/fetch_osm_boundaries.py` | Download Ukrainian boundaries from OSM (Cyrillic names) |
| `scripts/load_sectors.py` | Load the gazetteer, deriving colloquial aliases |
| `data/aliases.json` | Hand-maintained aliases for names the stem rules can't reach |
| `scripts/smoke_test.py` | End-to-end pipeline check, no API keys needed |
| `scripts/bench_payload.py` | Overlay/hot-path payload sizes vs. the client's timeouts |
| `scripts/verify_llm.py` | One-call check of the Gemini path (`--bisect` to debug schemas) |
| `data/sectors.sample.geojson` | **Sample bounding boxes only** — replace before real use |

## Setup

```bash
cp .env.example .env
```

Fill in `GEMINI_API_KEY` (from <https://aistudio.google.com/apikey>),
`TELEGRAM_API_ID`, `TELEGRAM_API_HASH` (from <https://my.telegram.org/apps>)
and `TELEGRAM_CHANNELS`.

Generate a Telegram session string — this is interactive, so run it on your own
machine, not in the container:

```bash
pip install telethon && TELEGRAM_API_ID=... TELEGRAM_API_HASH=... python scripts/login.py
```

Paste the printed `TELEGRAM_SESSION=...` into `.env`, then:

```bash
docker compose up --build -d
```

Download the boundaries (once — public Overpass mirrors are slow, so expect
minutes per layer; the script is resumable and skips layers already fetched):

```bash
python scripts/fetch_osm_boundaries.py oblasts kyiv_districts raions hromadas
```

Load them. `--replace` wipes the table first, so pass every layer in one call:

```bash
docker compose exec worker python scripts/load_sectors.py --replace data/osm_oblasts.geojson data/osm_kyiv_districts.geojson data/osm_raions.geojson
```

Verify. This exercises sector resolution, geometry construction, the dedup
upsert, the Redis snapshot, all-clear expiry and the janitor — everything except
the model call and Telegram. Re-run it after changing `geo.py`, `database.py`
or `cache.py`:

```bash
docker compose exec worker python scripts/smoke_test.py
```

```bash
curl -s 'http://localhost:8000/healthz'
```

```bash
curl -s 'http://localhost:8000/api/v1/threats/check?latitude=50.54&longitude=30.51'
```

```json
{
  "in_risk_zone": true,
  "risk_level": "high",
  "threat_type": "shahed",
  "eta_window": "6–11 min",
  "eta_minutes": 9,
  "sector_name": "Оболонський район",
  "origin_sector": "Чернігівська область",
  "source_refs": [4471, 4468, 4462],
  "geometry": null
}
```

Three of those are provenance rather than answer, and each is null/empty far
more often than not:

- `eta_minutes` is the raw number `eta_window` is rendered from. The window is
  a snapshot of the countdown at response time, so a client polling every 30 s
  would watch the string sit still and then jump; with the raw minutes it can
  tick between polls. Same reason `/threats/zones` carries it per Feature.
- `origin_sector` is the approach district the corridor in `geometry` was drawn
  from — the gazetteer's spelling of it, not the model's, and null when the
  extracted origin matched no sector. An unresolved origin drew no corridor, so
  there would be nothing on screen for it to explain.
- `source_refs` are the Telegram message ids behind the threat, newest first
  and capped at `database.MAX_SOURCE_REFS`. Not the LLM's
  `ExtractedThreat.source_refs`, which index one batching window and mean
  nothing once it is gone — these are the durable ids those indices resolved
  to, so a number on screen traces back to the message it came from. PostGIS
  keeps the full set; only the newest few go over the wire.

Add `&include_geometry=true` to get the matched `risk` polygon back with the
answer, as a GeoJSON geometry thinned the same way the overlay is. It is opt-in
because this response is `no-store` and polled continuously — a client drawing
a map should hold the ETag-cached overlay below instead and pay for geometry
once per snapshot rather than once per poll. The field is for callers that
never fetch the overlay at all: a widget, a Live Activity.

The whole active set as a GeoJSON `FeatureCollection`, two Features per threat
(`core` = districts actually reported, `risk` = core plus buffer and corridor):

```bash
curl -s 'http://localhost:8000/api/v1/threats/zones'
```

Cached per snapshot version and revalidated with a weak ETag, so the steady
state is a 304 with no body. Needs no coordinates — a client can hold the
overlay and answer "am I in a zone" locally without telling the server where
its user is.

## The hot path

`GET /api/v1/threats/check` performs **no network I/O**. Active threats in a
country-scale deployment number in the hundreds, so every API replica holds the
full geometry set in memory as a Shapely `STRtree` and answers from it. The
background refresher polls a Redis version counter every 2 seconds and rebuilds
the index only when it changes — steady state is one 4-byte `GET` per replica
per tick.

The trade is explicit: up to `SNAPSHOT_POLL_SECONDS` (2s) of staleness against a
threat TTL measured in tens of minutes, in exchange for removing the database
from the request path. Expiry is still enforced per request against the row's
`expires_at`, so a lapsed threat is never reported even if the snapshot is stale.

PostGIS remains the source of truth. If Redis is empty or unreachable, the API
falls back to `ST_Contains` against the GiST-indexed `geom` column
(`database.find_threats_at_point`) and keeps serving.

**Risk grading.** Each cluster carries two geometries: `core_geom` (the district
polygons — a direct hit) and `geom` (core + a threat-type-sized buffer + the
directional corridor from the approach sector). Inside `core_geom` is `high`,
inside `geom` only is `elevated`, and a `high` hit is downgraded to `elevated`
when the cluster's confidence is below `HIGH_CONFIDENCE`.

**Wire geometry is never lookup geometry.** Anything that leaves the process is
Douglas-Peucker thinned (topology-preserving) and capped at
`ZONES_MAX_VERTICES`. A fixed tolerance bounds a shape's error but not its
size, and the payload otherwise grows with how much of the country is under
alert: at 0.005° one Kyiv district is 21 vertices, a whole oblast is still 688.
The cap escalates the tolerance per shape until it fits. Measured on a
12-oblast worst case, 117,616 raw vertices ship as 3,310 — a 20.6 KiB gzipped
overlay, 5.5 s to last byte on 2G against the client's 8 s budget. See
`scripts/bench_payload.py`.

## Cost model

Four mechanisms, in order of impact:

1. **One call per window, not per message.** A 45-second window during a barrage
   might carry 40 messages and still cost one request.
2. **Keyword prefilter** (`worker._RELEVANCE_PATTERN`). Most traffic in a
   monitoring channel is not a threat report; anything that fails the regex
   never reaches the model. Tune it against your channels — a false negative
   here is a missed threat, so keep it generous.
3. **In-window deduplication.** Channels relay each other verbatim; identical
   texts are collapsed before the prompt is built.
4. **Prompt caching.** The long system prompt is byte-stable and sits in
   `system_instruction`, which is the very front of the request — exactly what
   Gemini's implicit caching keys on. Only the batched messages vary. Watch the
   `cached=` field in the `llm_window` log line: if it stays at zero across
   windows, either the prefix is too short to qualify or something is mutating
   it. If you need a guarantee rather than best-effort, switch to explicit
   caching (`client.caches.create`) — that is a change in `app/llm.py` only.

`LLM_THINKING_LEVEL=LOW` keeps reasoning spend down; this task is extraction,
not analysis. Model choice is the other lever you own —
`LLM_MODEL=gemini-3.5-flash-lite` is cheaper still for a task this constrained.
If you drop to the Gemini 2.5 family, set `LLM_THINKING_LEVEL=` (empty), because
those models use `thinking_budget` rather than `thinking_level`.

## Extraction contract

The model returns exactly the requested fields (`threat_type`, `count`,
`target_sectors`, `confidence_score`, `ttl_minutes`) plus four additions worth
flagging:

- `eta_minutes` — feeds the `eta_window` string the response requires.
- `origin_sector` — the approach direction, which is what makes a *corridor*
  possible rather than a plain radius.
- `source_refs` — batch indices grounding each threat, mapped back to real
  Telegram message ids for audit.
- `all_clear_sectors` — "відбій" announcements expire live threats immediately
  instead of waiting out the TTL.

The Pydantic model in `app/schemas.py` is passed to Gemini directly as
`response_schema`, so the response is schema-bound and arrives as a validated
instance on `response.parsed`. There is no regex extraction and no
retry-on-parse loop. Note that the `description=` on each field is not a comment
— Gemini receives it, so `schemas.py` is prompt surface as much as `llm.py` is.

**Safety filtering is the failure mode to watch.** This workload is entirely
munitions vocabulary, and at Gemini's default thresholds routine reports get
blocked — the system would go blind during exactly the windows that matter.
`LLM_SAFETY_THRESHOLD` defaults to `BLOCK_ONLY_HIGH` for that reason. Both
block paths are handled: a blocked *prompt* returns no candidates at all
(checked via `prompt_feedback.block_reason`), and a blocked *response* returns a
candidate with a `SAFETY`/`PROHIBITED_CONTENT` finish reason. Either raises
`LLMRefusal` and is logged as `llm_refused` — if that line starts appearing,
loosen the threshold.

`app/llm.py` translates every Gemini error into one of four provider-neutral
exceptions (`LLMRefusal`, `LLMExtractionError`, `LLMTransient`,
`LLMRequestError`) so the rest of the codebase has no idea which model vendor is
behind it. Swapping providers again is a one-file change.

## Deduplication and TTL

A cluster's identity is `sha1(threat_type + sorted normalised sector set)` —
count and confidence are excluded, because those are the fields that merge. On
re-report the row takes `GREATEST` of count and confidence, unions the sector
and message-id arrays, and pushes `expires_at` forward (never backward). A
`pg_advisory_xact_lock` on the dedup key serialises the read-then-write so two
overlapping windows cannot both insert.

Expiry is expressed twice on purpose: every read filters `expires_at > now()`
(correct the instant a threat lapses), and the janitor archives expired rows
into `threat_history` every 60 seconds so the working set — and the GiST index —
stays small.

## Sector resolution

Channels write district names inflected, abbreviated and misspelled. Resolution
is a three-step ladder: exact match on the normalised name → the sector's
`aliases` array → `pg_trgm` similarity above `SECTOR_MATCH_THRESHOLD`. A cluster
that resolves to nothing is **dropped**, not stored, and logged as
`threat_unlocalised` — an unlocalisable threat cannot answer a point query, and
a wrongly localised one sends people to shelter in the wrong city. Watch that
log line: it is your gazetteer-coverage metric.

Buffers are computed on the `geography` type, so 15 km is 15 km rather than a
degree-shaped ellipse.

### Where the boundary data comes from, and why

**It must carry Cyrillic names.** This is not a preference. The extraction model
emits Ukrainian (`Оболонський район`, `Київської області`), and romanised
gazetteers — geoBoundaries ships `Bakhchysarai`, `Simferopol` — cannot match
them: `similarity('оболонський','obolonska')` is exactly **0.000**, as is every
other cross-alphabet pair. A romanised gazetteer matches nothing at all.

`scripts/fetch_osm_boundaries.py` pulls from OSM, which carries the native
`name` tag, across four levels:

| Layer | OSM `admin_level` | Count |
|---|---|---|
| Oblasts | 4 | 27 |
| Raions | 6 | 161 |
| Kyiv city districts | **10** | 10 |
| Hromadas | 7 | 1467 |
| | | **1665 total** |

All four loaded at full precision: API resident memory 325 MB (from 227 MB with
oblasts+raions alone), startup 3 s, and the hot path unaffected at p50 0.97 ms /
p95 1.36 ms — the STRtree probe does not care how many polygons are in it.

Forty hromada names are not unique (`Піщанська сільська` appears four times).
They are all kept; the resolver's area tie-break picks the largest
deterministically. Watch `sector_unmatched` in the logs to see whether that
choice is ever wrong in practice.

Two things about that table are easy to get wrong. Kyiv's districts are
level **10**, not 9 — level 9 around Kyiv is the surrounding settlements
(Вишгород, Бровари). And Overpass `area(...)` scoping silently returned zero
elements on the public mirrors, so the raion and hromada queries use bounding
boxes and are trimmed to Ukraine locally against the oblast polygons (a bbox
over Ukraine also returns Belarusian and Polish raions — 265 of 426 were
dropped that way).

### Colloquial names

Channels rarely write the canonical form. `load_sectors.py` derives stem
aliases at load time because the colloquial forms score *below* the 0.45
threshold against the full name but well above it against the stem:

| Written | vs canonical name | vs derived stem |
|---|---|---|
| `Харківщина` | 0.375 ✗ | 0.500 ✓ (`харків`) |
| `Оболонь` | 0.429 ✗ | 0.667 ✓ (`оболон`) |

The `-щина` form is also generated explicitly, turning a fuzzy match into an
exact one. What stems cannot reach — irregular genitives (`Києва` scores 0.222
against `київ`) and neighbourhood names that are not administrative units at all
(`Троєщина` is inside Деснянський район) — lives in `data/aliases.json`. When a
real threat shows up in the logs as `sector_unmatched`, that file is where the
fix goes.

## Scaling

- **api** scales horizontally; each process holds its own index and refreshes
  independently. Scale on CPU.
- **worker** is intentionally single-instance. The batching window is stateful
  and the advisory-lock upsert is designed for low writer concurrency. To shard,
  split by channel group and give each worker its own set.

## Verified / not verified

The full stack has been built and run under Docker. Verified live against
PostGIS and Redis, via `scripts/smoke_test.py` and the HTTP endpoints:

- sector resolution across all three tiers — exact, alias, `pg_trgm` fuzzy —
  plus raw coordinates and correct rejection of unknown names;
- PostGIS geometry construction, including that the risk area strictly contains
  the core and that buffers are metres, not degrees (63 km² core → 1758 km²
  risk area with the corridor);
- the deduplicating upsert: a re-report updates rather than inserts, `count` and
  `confidence` take the maximum, source message ids accumulate;
- `ST_Contains` point lookup, snapshot publish/decode through Redis, all-clear
  expiry, and janitor archival into `threat_history`;
- all three risk levels over HTTP, `422` on an out-of-range latitude, the map
  endpoint, and ETA decay across real elapsed time;
- hot-path latency: **p50 1.14 ms, p95 1.65 ms** over 300 HTTP requests
  (host → Docker → uvicorn → handler);
- overlay payload budgets, via `scripts/bench_payload.py`: every shipped shape
  within the 200-vertex cap on loads from one district to twelve oblasts, and
  both the overlay and `include_geometry=true` inside the client's timeouts on
  every link profile down to 2G.

Four bugs were found this way that review alone had missed:

1. Shapely 2.1's `STRtree.query` applies the predicate as
   `input.predicate(tree_geometry)`, so `predicate="contains"` asked whether the
   *point* contained the polygon and matched nothing — every lookup returned "no
   threat". Now `intersects`, which is symmetric and counts the boundary as
   inside.
2. SQLAlchemy's `text()` does not recognise `:param::type` as a bind parameter
   (the name must not be followed by a colon). Nine statements would have failed
   at runtime; all casts are now `CAST(:param AS type)`.
3. pydantic-settings JSON-decodes list-typed fields straight from the
   environment, before any validator runs, so `TELEGRAM_CHANNELS=a,b` crashed
   both services at startup. Fixed with `Annotated[list[str], NoDecode]`.
4. `sector_name` returned whichever polygon the tree hit first. Because
   administrative boundaries nest, a user in Obolon was told "Київська область".
   `SectorIndex` now returns the smallest matching polygon.

Two smaller ones: scripts could not `import app` inside the container
(`PYTHONPATH=/srv` now set in the Dockerfile — this broke the documented
`load_sectors.py` command), and a missing `GEMINI_API_KEY` took the entire
worker down, janitor included, because the Gemini client was constructed
eagerly. It is lazy now, so a missing or rotated key disables extraction only.

**The whole chain is verified**, including the two paths that needed live
credentials. Telegram connects and subscribes; Gemini extraction was confirmed
against a realistic batch (two channels relaying one drone group, a correction,
a fundraising post and an all-clear). The model merged the duplicate reports
into a single threat, took the maximum count, ignored the fundraiser, routed the
all-clear to `all_clear_sectors`, and cited the supporting message indices.

Re-run that check any time — it costs one API call:

```bash
docker compose exec worker python scripts/verify_llm.py
```

Getting there required replacing the SDK's automatic Pydantic conversion with
`llm.gemini_response_schema()`. The SDK emits more than Gemini's accepted schema
subset (`additional_properties`, `title`, `default`, numeric bounds), and the API
rejects the surplus with a bare `400 INVALID_ARGUMENT / "Request contains an
invalid argument"` that names nothing. Our converter emits only `type`,
`description`, `enum`, `items`, `properties`, `required` and `nullable`. The
dropped constraints are still enforced client-side when the response is
validated back into the Pydantic model. If a future schema change gets rejected
again, `verify_llm.py --bisect` adds fields back one at a time and names the
offender.

**One thing the live run exposed:** the model returns Ukrainian place names in
grammatical case, not base form — `Київської області` (genitive), `Одеській
області` (locative), `Києва` (genitive). The resolution ladder handles this: the
noise-word stripping plus `pg_trgm` similarity matched both inflected oblast
names correctly, and `Оболонь` + `Оболонський район` deduplicated to a single
sector. `Києва` correctly went unmatched rather than falsely snapping to
"Київська область" — there is no Kyiv city polygon in the sample gazetteer.
That is the intended failure mode, and it disappears once real boundaries are
loaded.

Four gotchas worth knowing, all found the hard way:

* `docker compose restart` does **not** re-read `env_file`. After editing
  `.env`, use `docker compose up -d --force-recreate`.
* Model ids retire. `gemini-2.5-flash` returns 404 for keys created after its
  cutoff. `client.models.list()` shows what a given key can actually use.
* Class docstrings on the schema models become the `description` the model
  reads. Keep implementation commentary in `#` comments, not docstrings.
* Free-tier Gemini quota is per-day and easy to spend while debugging. The
  worker calls the API once per active window, so budget for it before relying
  on this in a real alert.

## Before production

- Replace `app/bootstrap.py` with Alembic migrations.
- Load real administrative boundaries.
- Put authentication and per-device rate limiting in front of `/api/v1/*`.
- Add metrics for `threat_unlocalised`, `llm_refused`, `ingest_queue_full` and
  cache hit rate — those four cover the failure modes that degrade quietly.
- Build an eval set of real windows with expected extractions before tuning the
  prompt; the system prompt is the highest-leverage file in the repo.
