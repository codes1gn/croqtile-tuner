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
    """Drop all tables and recreate from models on every startup.

    Disk artifacts (tuning/) are the source of truth.  The DB is a transient
    index rebuilt via scan_and_create_tasks / state_seed at startup.
    """
    from .models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    async with async_session() as session:
        yield session
