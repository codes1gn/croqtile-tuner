from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from .config import settings

_db_path_str = settings.db_path
if _db_path_str == ":memory:":
    _db_url = "sqlite+aiosqlite:///:memory:"
else:
    _db_path = Path(_db_path_str)
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _db_url = f"sqlite+aiosqlite:///{_db_path.resolve()}"

engine = create_async_engine(_db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    from .models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        columns = {
            row[1]
            for row in (await conn.exec_driver_sql("PRAGMA table_info(tasks)")).fetchall()
        }
        if "model" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN model VARCHAR(128)")
        if "opencode_session_id" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN opencode_session_id VARCHAR(128)")
        if "variant" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN variant VARCHAR(32) DEFAULT ''")
        if "dsl" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN dsl VARCHAR(32)")
        if "respawn_count" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN respawn_count INTEGER DEFAULT 0 NOT NULL")
        if "request_budget" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN request_budget INTEGER DEFAULT 1 NOT NULL")
        if "request_number" not in columns:
            await conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN request_number INTEGER DEFAULT 0 NOT NULL")
        iter_columns = {
            row[1]
            for row in (await conn.exec_driver_sql("PRAGMA table_info(iteration_logs)")).fetchall()
        }
        if "request_number" not in iter_columns:
            await conn.exec_driver_sql("ALTER TABLE iteration_logs ADD COLUMN request_number INTEGER")

        await conn.exec_driver_sql(
            "DELETE FROM iteration_logs WHERE id NOT IN ("
            "  SELECT MIN(id) FROM iteration_logs GROUP BY task_id, iteration"
            ")"
        )


async def get_session():
    async with async_session() as session:
        yield session
