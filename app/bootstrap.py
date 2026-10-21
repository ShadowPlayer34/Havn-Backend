"""Schema bootstrap.

Extensions (postgis, pg_trgm) are created by the container init script; tables
are created from the SQLAlchemy metadata. For a real deployment, swap this for
Alembic — it is here so `docker compose up` works from an empty volume.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings
from app.database import Base, create_engine

logger = logging.getLogger(__name__)


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
        # `create_all` only ever creates whole tables — a column added to a
        # model after the table exists is silently missing on any volume that
        # predates it. Until this grows an Alembic history, additive columns go
        # here so an already-running deployment picks them up on restart.
        await conn.execute(
            text(
                "ALTER TABLE active_threats "
                "ADD COLUMN IF NOT EXISTS origin_sector varchar(160)"
            )
        )
        # Trigram index for the fuzzy district-name fallback in geo.py.
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_sectors_name_norm_trgm "
                "ON sectors USING gin (name_norm gin_trgm_ops)"
            )
        )
    logger.info("schema_ready")


async def _main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    engine = create_engine(get_settings())
    try:
        await ensure_schema(engine)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
