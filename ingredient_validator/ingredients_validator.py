import asyncio
import platform
from pathlib import Path
from typing import Any, Dict, Set

import aiohttp
import httpx
import polars as pl
import stamina
from loguru import logger
from openfoodfacts import Lang
from tqdm import tqdm

from .utils import (
    FileManager,
    retry_only_on_real_errors,
    retry_only_on_real_errors_aiohttp,
    timeit,
)


class IngredientValidator:
    def __init__(self, lang: Lang, cache_dir: Path = Path("cache")):
        self.lang = lang
        self.timeout = 90
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.url = (
            f"https://{self.lang.value}.openfoodfacts.org/ingredients"
            f".json?status=unknown"
        )
        self.user_agent = {"User-Agent": f"MyApp/{platform.python_version()}"}
        self.unknown_ingredients_fp = (
            self.cache_dir / f"0_unknown_ingredients_{self.lang.value}.json"
        )
        self.processed_ingredients_fp = (
            self.cache_dir / f"1_processed_ingredients_{self.lang.value}.json"
        )
        self.validated_ingredients_fp = (
            self.cache_dir / f"2_dwds_validated_ingredients_{self.lang.value}.json"
        )
        self.filtered_ingredients = None
        self.unknown_ingredients = None
        self.file_manager = FileManager()
        self.max_url_length = 2048
        self.dwds_api_url = "https://www.dwds.de/api/wb/snippet/?q="

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
        logger.info("Returning preprocessed ingredients")
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

    @timeit
    def validate_ingredients_with_dwds(
        self, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Validate ingredients using the DWDS API, processing only unvalidated ingredients."""
        logger.info("Validating ingredients with DWDS API...")

        # Load validated ingredients from the cache if it's up-to-date and not forced to refresh
        if not force_refresh and self.file_manager.is_file_current(
            self.validated_ingredients_fp
        ):
            logger.info("Loading validated ingredients from cache.")
            validated_data = self.file_manager.load_json(self.validated_ingredients_fp)

            # Extract validated ingredient IDs from both valid and invalid lists
            checked_already = set().union(*validated_data.values())

            # Find unvalidated ingredients (those not in the validated file)
            delta = set(self.filtered_ingredients) - checked_already

            if delta:
                logger.info(f"Validating {len(delta)} unvalidated ingredients.")
                validated_delta = asyncio.run(self._async_validate_words(delta))

                # Merge the newly validated ingredients with the existing ones
                validated_data["valid"].extend(
                    validated_delta["valid"]
                )  # Add validated ingredients
                validated_data["invalid"].extend(
                    validated_delta["invalid"]
                )  # Add invalid ingredients

                # Save the updated validated ingredients back to the file
                self.file_manager.save_json(
                    self.validated_ingredients_fp, validated_data
                )
                self.validated_ingredients = validated_data
            else:
                logger.info(
                    "No new ingredients to validate. All ingredients are already validated."
                )

            return validated_data

        # If the cache is not current or force_refresh is True, validate all ingredients
        logger.info(f"Validating {len(self.filtered_ingredients)} ingredients.")
        validated_data = asyncio.run(
            self._async_validate_words(self.filtered_ingredients)
        )

        # Save the newly validated ingredients to the file
        self.file_manager.save_json(self.validated_ingredients_fp, validated_data)
        self.validated_ingredients = validated_data
        logger.info(
            f"Validation complete. Valid: {len(validated_data['valid'])}, "
            f"Invalid: {len(validated_data['invalid'])}"
        )
        return validated_data

    async def _async_validate_words(self, words: Set[str]) -> Dict[str, list]:
        batches = self._calculate_batches(words)

        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    self._fetch_batch(session, self._build_url(batch), batch)
                )
                for batch in batches
            ]

            validated_results = []
            with tqdm(total=len(tasks), desc="Validating ingredients") as pbar:
                for coroutine in asyncio.as_completed(tasks):
                    result = await coroutine
                    validated_results.append(result)
                    pbar.update(1)

            valid, invalid = self._aggregate_results(validated_results, words)
            return {"valid": sorted(valid), "invalid": sorted(invalid)}

    @timeit
    def _calculate_batches(self, words):
        base_length = len(self.dwds_api_url)
        batches = []
        current_batch = []
        current_length = base_length
        for word in words:
            contribution = len(word) + 1
            if current_length + contribution > self.max_url_length:
                batches.append(current_batch)
                current_batch = [word]
                current_length = base_length + len(word)
            else:
                current_batch.append(word)
                current_length += contribution
        if current_batch:
            batches.append(current_batch)
        return batches

    def _build_url(self, batch: list) -> str:
        return f"{self.dwds_api_url}{'|'.join(batch)}"

    def _aggregate_results(self, results: list, original_words: Set[str]) -> tuple:
        valid = set()
        invalid = set()
        for res in results:
            valid.update(res["valid"])
            invalid.update(res["invalid"])
        # Ensure all words are accounted for
        assert len(valid) + len(invalid) == len(
            original_words
        ), "Mismatch in validation results."
        return valid, invalid

    @stamina.retry(on=retry_only_on_real_errors_aiohttp, attempts=3)
    async def _fetch_batch(
        self, session: aiohttp.ClientSession, url: str, batch: list
    ) -> Dict[str, set]:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                valid_inputs = {item["input"] for item in data}
                return {
                    "valid": set(batch) & valid_inputs,
                    "invalid": set(batch) - valid_inputs,
                }
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                logger.error(f"Client error for {url}: {e}")
                return {"valid": set(), "invalid": set(batch)}
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Network error for {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            return {"valid": set(), "invalid": set(batch)}
