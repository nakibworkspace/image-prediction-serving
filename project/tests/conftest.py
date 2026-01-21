# project/tests/conftest.py

import os

import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import Settings, get_settings
from app.main import create_application
from app.db import Base, get_session


def get_settings_override():
    return Settings(testing=1, database_url=os.environ.get("DATABASE_TEST_URL"))


@pytest.fixture(scope="module")
def test_app():
    # set up
    app = create_application()
    app.dependency_overrides[get_settings] = get_settings_override
    with TestClient(app) as test_client:
        # testing
        yield test_client
    # tear down


@pytest.fixture(scope="module")
def test_app_with_db():
    # set up
    app = create_application()
    app.dependency_overrides[get_settings] = get_settings_override

    # Get test database URL and convert for async
    DATABASE_TEST_URL = os.environ.get("DATABASE_TEST_URL", "sqlite+aiosqlite:///./test.db")
    if DATABASE_TEST_URL.startswith("postgres://"):
        DATABASE_TEST_URL = DATABASE_TEST_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif DATABASE_TEST_URL.startswith("sqlite:///") and "+aiosqlite" not in DATABASE_TEST_URL:
        DATABASE_TEST_URL = DATABASE_TEST_URL.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

    # Create test engine and session
    test_engine = create_async_engine(DATABASE_TEST_URL)
    test_session_maker = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session():
        async with test_session_maker() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as test_client:
        # testing
        yield test_client
    # tear down
