import functools
from timeit import default_timer as timer
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable

import aiohttp
import httpx
import orjson
from loguru import logger


def timeit(func: Callable) -> Callable:
    """Decorator to log function execution time."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        duration = timer() - start
        logger.debug(f"'{func.__name__}' executed in {duration:.3f}s")
        return result

    return wrapped


def retry_only_on_real_errors(exc: Exception) -> bool:
    """Retry only on server errors (5xx) or network-related issues."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.HTTPError)


def retry_only_on_real_errors_aiohttp(exc: Exception) -> bool:
    """Retry only on server errors (5xx) or network-related issues for aiohttp."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, aiohttp.ClientError)


class FileManager:
    """Handles file operations with caching capabilities."""

    @staticmethod
    def is_file_current(filepath: Path) -> bool:
        """Check if the cached file exists and was modified today."""
        if not filepath.exists():
            return False
        cache_date = datetime.fromtimestamp(filepath.stat().st_mtime).date()
        return cache_date == datetime.today().date()

    @staticmethod
    def load_json(filepath: Path) -> Dict[str, Any]:
        """Load JSON data from a file."""
        return orjson.loads(filepath.read_bytes()) if filepath.exists() else {}

    @staticmethod
    def save_json(filepath: Path, data: Dict[str, Any]) -> None:
        """Save JSON data to a file."""
        with open(filepath, "wb") as f:
            f.write(orjson.dumps(data))
