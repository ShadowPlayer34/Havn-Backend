"""One-time Telegram login. Prints a StringSession for TELEGRAM_SESSION.

Run on a machine where you can receive the Telegram code:

    python scripts/login.py

Telethon writes nothing to disk in this mode — copy the printed string into
`.env` as TELEGRAM_SESSION and it is the only credential the worker needs.
Treat it like a password: it is a full login to that account.
"""

from __future__ import annotations

import asyncio
import os
import sys

from telethon import TelegramClient
from telethon.sessions import StringSession


async def main() -> int:
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        print(
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH first "
            "(create an app at https://my.telegram.org/apps).",
            file=sys.stderr,
        )
        return 1

    async with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        me = await client.get_me()
        print(f"\nSigned in as: {me.username or me.id}")
        print("\nTELEGRAM_SESSION=" + client.session.save())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
