from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from .config import AVAILABLE_VARIANTS, fetch_all_models, settings
from .models import SystemSetting

DEFAULT_MODEL_KEY = "default_model"
DEFAULT_VARIANT_KEY = "default_variant"
AUTO_WAKE_KEY = "auto_wake_enabled"
USE_PROXY_KEY = "use_proxy"


def available_models() -> list[str]:
    return fetch_all_models()


def available_variants() -> list[str]:
    return list(AVAILABLE_VARIANTS)


async def get_default_model(session: AsyncSession) -> str:
    row = await session.get(SystemSetting, DEFAULT_MODEL_KEY)
    if row is None:
        row = SystemSetting(key=DEFAULT_MODEL_KEY, value=settings.opencode_model)
        session.add(row)
        await session.flush()
    return row.value


async def get_default_variant(session: AsyncSession) -> str:
    row = await session.get(SystemSetting, DEFAULT_VARIANT_KEY)
    if row is None:
        row = SystemSetting(key=DEFAULT_VARIANT_KEY, value=settings.opencode_variant)
        session.add(row)
        await session.flush()
    return row.value


async def set_default_model(session: AsyncSession, model: str, variant: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)

    m_row = await session.get(SystemSetting, DEFAULT_MODEL_KEY)
    if m_row is None:
        m_row = SystemSetting(key=DEFAULT_MODEL_KEY, value=model, updated_at=now)
        session.add(m_row)
    else:
        m_row.value = model
        m_row.updated_at = now

    v_row = await session.get(SystemSetting, DEFAULT_VARIANT_KEY)
    if v_row is None:
        v_row = SystemSetting(key=DEFAULT_VARIANT_KEY, value=variant, updated_at=now)
        session.add(v_row)
    else:
        v_row.value = variant
        v_row.updated_at = now

    await session.flush()
    return m_row.value, v_row.value


async def get_auto_wake_enabled(session: AsyncSession) -> bool:
    """Get the auto-wake toggle state. Default is False (monitor-only mode)."""
    row = await session.get(SystemSetting, AUTO_WAKE_KEY)
    if row is None:
        row = SystemSetting(key=AUTO_WAKE_KEY, value="false")
        session.add(row)
        await session.flush()
    return row.value.lower() == "true"


async def set_auto_wake_enabled(session: AsyncSession, enabled: bool) -> bool:
    """Set the auto-wake toggle state."""
    now = datetime.now(timezone.utc)
    row = await session.get(SystemSetting, AUTO_WAKE_KEY)
    value = "true" if enabled else "false"
    if row is None:
        row = SystemSetting(key=AUTO_WAKE_KEY, value=value, updated_at=now)
        session.add(row)
    else:
        row.value = value
        row.updated_at = now
    await session.flush()
    return enabled


async def get_use_proxy(session: AsyncSession) -> bool:
    """Get the proxy toggle state. Default is False."""
    row = await session.get(SystemSetting, USE_PROXY_KEY)
    if row is None:
        row = SystemSetting(key=USE_PROXY_KEY, value="false")
        session.add(row)
        await session.flush()
    return row.value.lower() == "true"


async def set_use_proxy(session: AsyncSession, enabled: bool) -> bool:
    """Set the proxy toggle state."""
    now = datetime.now(timezone.utc)
    row = await session.get(SystemSetting, USE_PROXY_KEY)
    value = "true" if enabled else "false"
    if row is None:
        row = SystemSetting(key=USE_PROXY_KEY, value=value, updated_at=now)
        session.add(row)
    else:
        row.value = value
        row.updated_at = now
    await session.flush()
    return enabled
