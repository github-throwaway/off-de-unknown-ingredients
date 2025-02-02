import functools
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Set

import httpx
import orjson
import polars as pl
from icecream import ic
from loguru import logger
from openfoodfacts import Lang
import stamina


def timeit(func):
    """Decorator to log function execution time."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(
            "'{name}' executed in {duration:.3f}s",
            name=func.__name__,
            duration=duration,
        )
        return result

    return wrapped


def retry_only_on_real_errors(exc: Exception) -> bool:
    # If the error is an HTTP status error, only retry on 5xx errors.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500

    # Otherwise retry on all httpx errors.
    return isinstance(exc, httpx.HTTPError)


class IngredientValidator:
    def __init__(self, lang: Lang, cache_dir: Path = Path("cache")):
        self.lang = lang
        self.timeout = 90
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.user_agent = {"User-Agent": f"MyAwesomeApp/{platform.python_version()}"}
        self.unknown_ingredients_filepath = (
            self.cache_dir / f"unknown_ingredients_{self.lang.value}.json"
        )
        self.url = (
            f"https://{self.lang.value}.openfoodfacts.org/ingredients.json"
            f"?status=unknown"
        )
        self.unknown_ingredients = None

    @timeit
    def preprocess_ingredients(self, data: dict) -> Set[str]:
        """Load and process ingredients from a JSON file using Polars."""
        df = pl.DataFrame(data["tags"])

        logger.info(f"Loaded {df.height} unknown ingredients.")

        filtered_df = df.filter(
            pl.col("id").str.contains(f"{self.lang.value}:")
            & ~pl.col("id").str.contains("-")
            & (pl.col("name").str.len_chars() > 1)
            & pl.col("name").str.contains(r"^[A-Za-zäöüßÄÖÜ]+$")
        )

        filtered_ingredients = filtered_df.get_column("name").to_list()

        logger.info(
            f"Reduced to {len(filtered_ingredients)} one-word strings with "
            f"latin characters and German as input language."
        )
        return set(filtered_ingredients)

    def _is_file_current(self) -> bool:
        """
        Check if the cached file exists and was modified today.
        """
        if not self.unknown_ingredients_filepath.exists():
            return False

        cache_date = datetime.fromtimestamp(
            self.unknown_ingredients_filepath.stat().st_mtime
        ).date()
        return cache_date == datetime.today().date()

    @stamina.retry(on=retry_only_on_real_errors, attempts=3)
    @timeit
    def _download_and_cache_unknown_ingredients(self) -> None:
        """
        Download ingredients data and cache it locally.
        """
        logger.info(f"Downloading ingredients for {self.lang.value} from {self.url}")
        resp = httpx.get(self.url, timeout=self.timeout, headers=self.user_agent)
        resp.raise_for_status()
        with open(self.unknown_ingredients_filepath, "wb") as f:
            f.write(orjson.dumps(resp.json()))

    @timeit
    def fetch_unknown_ingredients(self) -> dict:
        """
        Fetch ingredients data using caching.
        If the cache is outdated, download new data.

        Returns:
            The ingredients data as a dictionary.
        """
        if not self._is_file_current():
            try:
                self._download_and_cache_unknown_ingredients()
            except (httpx.HTTPError, IOError) as e:
                if self.unknown_ingredients_filepath.exists():
                    logger.warning("Using cached data due to error: {}", e)
                else:
                    raise

        self.unknown_ingredients = orjson.loads(
            self.unknown_ingredients_filepath.read_bytes()
        )
        return self.unknown_ingredients


@timeit
def main():
    fetcher = IngredientValidator(Lang.de)
    data = fetcher.fetch_unknown_ingredients()

    data = fetcher.preprocess_ingredients(data)
    ic(data)


if __name__ == "__main__":
    main()
