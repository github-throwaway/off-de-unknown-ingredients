import platform
from pathlib import Path
from typing import Set, Dict, Any

import httpx
import polars as pl
from loguru import logger
from openfoodfacts import Lang
import stamina

from .utils import timeit, retry_only_on_real_errors, FileManager


class IngredientValidator:
    def __init__(self, lang: Lang, cache_dir: Path = Path("cache")):
        self.lang = lang
        self.timeout = 90
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.url = f"https://{self.lang.value}.openfoodfacts.org/ingredients.json?status=unknown"
        self.user_agent = {"User-Agent": f"MyApp/{platform.python_version()}"}
        self.unknown_ingredients_fp = (
            self.cache_dir / f"0_unknown_ingredients_{self.lang.value}.json"
        )
        self.processed_ingredients_fp = (
            self.cache_dir / f"1_processed_ingredients_{self.lang.value}.json"
        )
        self.filtered_ingredients = None
        self.unknown_ingredients = None
        self.file_manager = FileManager()

    @timeit
    def preprocess_ingredients(self, force_refresh: bool = False) -> Set[str]:
        """Load and process ingredients from JSON using Polars."""
        if force_refresh or not self.file_manager.is_file_current(
            self.unknown_ingredients_fp
        ):
            logger.info("Processing ingredients...")
            df = pl.DataFrame(self.unknown_ingredients.get("tags", []))
            logger.info(f"Loaded {df.height} unknown ingredients.")

            filtered_df = df.filter(
                pl.col("id").str.contains(f"{self.lang.value}:")
                & ~pl.col("id").str.contains("-")
                & (pl.col("name").str.len_chars() > 1)
                & pl.col("name").str.contains(r"^[A-Za-zäöüßÄÖÜ]+$")
            )

            filtered_ingredients = filtered_df.get_column("name").to_list()
            logger.info(f"Reduced to {len(filtered_ingredients)} possible ingredients.")
        self.filtered_ingredients = set(
            self.file_manager.load_json(self.processed_ingredients_fp)
        )
        return self.filtered_ingredients

    @stamina.retry(on=retry_only_on_real_errors, attempts=3)
    @timeit
    def _download_and_cache_unknown_ingredients(self) -> None:
        """Download and cache unknown ingredients."""
        logger.info(f"Downloading ingredients for {self.lang.value} from {self.url}")
        resp = httpx.get(self.url, timeout=self.timeout, headers=self.user_agent)
        resp.raise_for_status()
        self.file_manager.save_json(self.unknown_ingredients_fp, resp.json())

    @timeit
    def fetch_unknown_ingredients(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch unknown ingredients, using cache when possible."""
        logger.info("Fetching unknown ingredients...")
        if force_refresh or not self.file_manager.is_file_current(
            self.unknown_ingredients_fp
        ):
            try:
                self._download_and_cache_unknown_ingredients()
            except (httpx.HTTPError, IOError) as e:
                if self.unknown_ingredients_fp.exists():
                    logger.warning("Using cached data due to error: {}", e)
                else:
                    raise
        self.unknown_ingredients = self.file_manager.load_json(
            self.unknown_ingredients_fp
        )
        return self.unknown_ingredients
