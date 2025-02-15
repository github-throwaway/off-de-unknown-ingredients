import asyncio
import math
import platform
import re
from collections import OrderedDict
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import aiohttp
import httpx
import polars as pl
import stamina
from bs4 import BeautifulSoup
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
        self.corpus = None
        self.validated_ingredients = None
        self.leipzig_semaphore = asyncio.Semaphore(50)
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
        self.leipzig_fp_template = (
            self.cache_dir / f"3_leipzig_{{corpus}}_{self.lang.value}.json"
        )
        self.openthesaurus_fp = (
            self.cache_dir / "4_openthesaurus_Gastronomie_Kulinarik.json"
        )

    def _get_leipzig_fp(self, corpus: str) -> Path:
        return Path(str(self.leipzig_fp_template).format(corpus=corpus))

    @timeit
    def preprocess_ingredients(self, force_refresh: bool = False) -> Set[str]:
        """Load and process ingredients from JSON using Polars."""
        if force_refresh or not self.file_manager.is_file_current(
            self.processed_ingredients_fp
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

            self.file_manager.save_json(
                self.processed_ingredients_fp, filtered_ingredients
            )
            self.filtered_ingredients = set(filtered_ingredients)
        else:
            self.filtered_ingredients = set(
                self.file_manager.load_json(self.processed_ingredients_fp)
            )

        logger.info(
            f"Reduced to {len(self.filtered_ingredients)} one-word ingredients."
        )
        return self.filtered_ingredients

    @stamina.retry(on=retry_only_on_real_errors, attempts=3)
    @timeit
    def _download_and_cache_unknown_ingredients(self) -> None:
        """Download and cache unknown ingredients."""
        logger.info(
            f"Downloading unknown ingredients for {self.lang.value} from {self.url}"
        )
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
                self.unknown_ingredients = self.file_manager.load_json(
                    self.unknown_ingredients_fp
                )
            except (httpx.HTTPError, IOError) as e:
                if self.unknown_ingredients_fp.exists():
                    logger.warning("Using cached data due to error: {}", e)
                    self.unknown_ingredients = self.file_manager.load_json(
                        self.unknown_ingredients_fp
                    )
                else:
                    raise
        else:
            if not self.unknown_ingredients:
                self.unknown_ingredients = self.file_manager.load_json(
                    self.unknown_ingredients_fp
                )

        logger.info(
            f"Fetched {len(self.unknown_ingredients['tags'])} unknown ingredients."
        )
        return self.unknown_ingredients

    @timeit
    def validate_ingredients_with_dwds(
        self, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Validate ingredients using the DWDS API."""
        logger.info("Validating ingredients with DWDS API...")

        if not force_refresh and self.file_manager.is_file_current(
            self.validated_ingredients_fp
        ):
            validated_data = self.file_manager.load_json(self.validated_ingredients_fp)
            checked_already = set().union(*validated_data.values())
            delta = set(self.filtered_ingredients) - checked_already

            if delta:
                validated_delta = asyncio.run(self._async_validate_words(delta))
                validated_data["valid"].extend(validated_delta["valid"])
                validated_data["invalid"].extend(validated_delta["invalid"])
                self.validated_ingredients = validated_data
                self.file_manager.save_json(
                    self.validated_ingredients_fp, validated_data
                )
        else:
            validated_data = asyncio.run(
                self._async_validate_words(self.filtered_ingredients)
            )
            self.file_manager.save_json(self.validated_ingredients_fp, validated_data)
        self.validated_ingredients = validated_data

        return self.validated_ingredients

    async def _async_validate_words(self, words: Set[str]) -> Dict[str, list]:
        batches = self._calculate_dwds_batches(words)

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
    def _calculate_dwds_batches(self, words):
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

    @staticmethod
    def _aggregate_results(results: list, original_words: Set[str]) -> tuple:
        valid = set()
        invalid = set()
        for res in results:
            valid.update(res["valid"])
            invalid.update(res["invalid"])
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

    async def fetch_primary(
        self, session: aiohttp.ClientSession, corpus: str, word: str
    ) -> List[str]:
        """Fetch groups using the primary endpoint."""
        primary_url = (
            f"https://corpora.uni-leipzig.de/de/webservice/index"
            f"?corpusId={corpus}&action=loadWordSetBox&word={word}"
        )
        async with session.get(primary_url, timeout=self.timeout) as response:
            response.raise_for_status()
            primary_text = await response.text()
            groups = re.findall(r"\d+\.\d+ [A-Za-zäöüßÄÖÜ, ]+", primary_text)
        return groups

    async def fetch_fallback(
        self, session: aiohttp.ClientSession, corpus: str, word: str
    ) -> List[str]:
        secondary_url = (
            f"https://corpora.uni-leipzig.de/de/res?corpusId={corpus}&word={word}"
        )

        async with session.get(secondary_url, timeout=self.timeout) as get_response:
            get_response.raise_for_status()
            secondary_text = await get_response.text()
            match = re.search(
                r"<b>Sachgebiet:</b>\s*([A-Za-zäöüßÄÖÜ\-, ]+)<br/>", secondary_text
            )
            return (
                [topic.strip() for topic in match.group(1).split(",")] if match else []
            )

    async def _fetch_dornseiff_entry(
        self,
        session: aiohttp.ClientSession,
        corpus: str,
        word: str,
        fallback: bool = True,
    ) -> Tuple[str, List[str]]:
        try:
            groups_primary = await self.fetch_primary(session, corpus, word)
        except Exception as e:
            logger.debug(f"Primary call failed for '{word}': {e!r}")
            groups_primary = []

        if groups_primary:
            return word, groups_primary

        if fallback:
            try:
                groups_fallback = await self.fetch_fallback(session, corpus, word)
            except Exception as e:
                logger.debug(f"Fallback call failed for '{word}': {e!r}")
                groups_fallback = []
            return word, groups_fallback

        return word, groups_primary

    @timeit
    def fetch_dornseiff_bedeutungsgruppe(
        self,
        corpus: str = "deu_news_2024",
        force_refresh: bool = False,
        fallback: bool = True,
    ) -> Dict[str, List[str]]:
        logger.info(f"Fetching Dornseiff-Bedeutungsgruppen from {corpus} corpus")
        self.corpus = corpus
        leipzig_fp = self._get_leipzig_fp(corpus)

        cached_results = {}
        if not force_refresh:
            cached_results = self.file_manager.load_json(leipzig_fp)

        valid_words = set(self.validated_ingredients["valid"])
        delta_words = {word for word in valid_words if word not in cached_results}

        if delta_words:

            async def process_entries():
                async with aiohttp.ClientSession(
                    headers=self.user_agent,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as session:
                    sem = asyncio.Semaphore(50)

                    async def fetch_word(word: str) -> Tuple[str, List[str]]:
                        async with sem:
                            return await self._fetch_dornseiff_entry(
                                session, corpus, word, fallback
                            )

                    tasks = [fetch_word(word) for word in delta_words]
                    results = {}

                    with tqdm(
                        total=len(tasks), desc=f"Fetching Leipzig groups for {corpus}"
                    ) as pbar:
                        for future in asyncio.as_completed(tasks):
                            word, groups = await future
                            results[word] = groups
                            pbar.update(1)
                    return results

            new_results = asyncio.run(process_entries())
            cached_results.update(new_results)
            self.file_manager.save_json(leipzig_fp, cached_results)

        return cached_results

    @timeit
    def fetch_openthesaurus_category(
        self, category: str, force_refresh: bool = False
    ) -> dict:
        if not force_refresh and self.file_manager.is_file_current(
            self.openthesaurus_fp
        ):
            logger.info("Loading OpenThesaurus terms from cache...")
            return self.file_manager.load_json(self.openthesaurus_fp)
        result = asyncio.run(self._async_scrape_openthesaurus(category))
        self.file_manager.save_json(self.openthesaurus_fp, result)
        return result

    @staticmethod
    async def _async_scrape_openthesaurus(category: str) -> dict:
        async def fetch_page(session, category, offset=0):
            params = {"category": category, "offset": offset, "noLevel": 0, "level": 5}

            @stamina.retry(on=aiohttp.ClientError, wait_initial=1.0, wait_max=10.0)
            async def fetch():
                async with session.get(
                    "https://www.openthesaurus.de/search/search", params=params
                ) as response:
                    response.raise_for_status()
                    return await response.text()

            return BeautifulSoup(await fetch(), "html.parser")

        async with aiohttp.ClientSession() as session:
            soup = await fetch_page(session, category)
            header = soup.find("h2")
            if header:
                total_str = header.text.split()[0].replace(".", "")
                try:
                    total = int(total_str)
                except ValueError:
                    total = 0
                    logger.warning(
                        "Unable to parse total count from header: %s", header.text
                    )
            else:
                total = 0
                logger.warning("No header found to determine total results.")

            all_results = [
                item.find("a").text
                for item in soup.select("#powerSearchResult ul li")
                if item.find("a")
            ]
            expected_pages = math.ceil(total / 20) if total > 0 else 1
            offsets = [i * 20 for i in range(1, expected_pages)]
            sem = asyncio.Semaphore(50)

            async def process_offset(offset: int) -> list:
                async with sem:
                    page_soup = await fetch_page(session, category, offset)
                    return [
                        item.find("a").text
                        for item in page_soup.select("#powerSearchResult ul li")
                        if item.find("a")
                    ]

            tasks = [process_offset(offset) for offset in offsets]
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Scraping OpenThesaurus",
            ):
                all_results.extend(await future)

        unique_terms = set(all_results)
        if total and (len(unique_terms) != total):
            logger.warning(
                f"Delta detected: expected {total} unique entries, but scraped {len(unique_terms)} unique entries."
            )
        else:
            logger.info("Scraped all %d entries successfully.", len(unique_terms))
        return {term: category for term in unique_terms}

    @timeit
    def combine_opent_and_dornseiff(self) -> dict:
        leipzig_cache = self._get_leipzig_fp(self.corpus)
        dornseiff_bedeutungsgruppe = self.file_manager.load_json(leipzig_cache)
        openthesaurus_results = self.file_manager.load_json(self.openthesaurus_fp)
        for word, category in openthesaurus_results.items():
            if (
                word in dornseiff_bedeutungsgruppe
                and category not in dornseiff_bedeutungsgruppe[word]
            ):
                dornseiff_bedeutungsgruppe[word].append(category)
        self.file_manager.save_json(leipzig_cache, dornseiff_bedeutungsgruppe)
        return dornseiff_bedeutungsgruppe

    @timeit
    def get_product_count(self, ingredients: List[str]) -> OrderedDict:
        """Returns a dict of ingredient counts, sorted in descending order."""
        prefix = f"{self.lang.value}:"
        ingredient_lookup = {f"{prefix}{ing.lower()}" for ing in ingredients}

        counts = {
            tag["id"].lower(): tag["products"]
            for tag in self.unknown_ingredients["tags"]
            if tag["id"].lower() in ingredient_lookup
        }

        return OrderedDict(sorted(counts.items(), key=itemgetter(1), reverse=True))
