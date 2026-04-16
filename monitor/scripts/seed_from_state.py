#!/usr/bin/env python3
"""
Import tasks from tuning/state.json (and optional results.tsv rows into iteration_logs).

Run from project root after copying croktile_paper/tuning -> CroqTuner/tuning:

  cd /path/to/CroqTuner/backend && source .venv/bin/activate
  python ../scripts/seed_from_state.py

Environment: same as backend (CROQTUNER_DB_PATH, etc.).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Run as if from backend/
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)


async def main() -> None:
    from sqlalchemy import delete, func, select

    from app.config import settings
    from app.database import async_session, engine, init_db
    from app.models import Base, IterationLog, Task
    from app.state_seed import seed_tasks_from_state_if_empty

    from app.state_seed import _find_state_files

    state_paths = _find_state_files(settings.tuning_dir)
    if not state_paths:
        print(f"No state.json found under {settings.tuning_dir}", file=sys.stderr)
        sys.exit(1)

    await init_db()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        await session.execute(delete(IterationLog))
        await session.execute(delete(Task))
        await session.commit()
    async with async_session() as session:
        seeded = await seed_tasks_from_state_if_empty(session)
        if not seeded:
            print(f"No tasks seeded from {settings.tuning_dir}", file=sys.stderr)
            sys.exit(1)

        task_count = (await session.execute(select(func.count(Task.id)))).scalar_one()
        log_count = (await session.execute(select(func.count(IterationLog.id)))).scalar_one()

    print(
        f"Seeded {task_count} tasks and {log_count} iteration logs "
        f"from {settings.tuning_dir}."
    )


if __name__ == "__main__":
    asyncio.run(main())
