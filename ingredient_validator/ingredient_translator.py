import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from git import Repo
from googletrans import Translator
from loguru import logger
from openfoodfacts import Lang

from .utils import FileManager, timeit


class IngredientTranslator:
    def __init__(
        self, repo_path: Union[Path, str], lang_from: str = "de", lang_to: str = "en"
    ):
        self.repo_path = Path(repo_path)  # Ensure repo_path is a Path object.
        self.taxonomy_path = self.repo_path / "taxonomies" / "food" / "ingredients.txt"
        self.lang_from = lang_from
        self.lang_to = lang_to
        self.repo = self._init_repo()
        # Translation cache stored in the project-level "cache" folder with "6_" prefix.
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.translations_cache_fp = (
            self.cache_dir / f"6_translations_{self.lang_from}_{self.lang_to}.json"
        )

    def _init_repo(self) -> Repo:
        """Initialize or clone the repository if it doesn't exist."""
        if not self.repo_path.exists():
            logger.info("Cloning OpenFoodFacts repository...")
            return Repo.clone_from(
                "https://github.com/openfoodfacts/openfoodfacts-server.git",
                self.repo_path,
            )
        return Repo(self.repo_path)

    def update_repo(self):
        """Pull latest changes from remote repository."""
        logger.info("Pulling latest changes...")
        self.repo.remotes.origin.pull()

    def load_taxonomy(self, lang: str) -> Set[str]:
        """Load taxonomy terms for a specific language."""
        terms = set()
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{lang}:"):
                    words = line[len(f"{lang}:") :].strip().lower().split(", ")
                    terms.update(words)
        return terms

    @timeit
    async def translate_terms(
        self, terms: List[str], force_refresh: bool = False
    ) -> Dict[str, str]:
        """
        Translate terms in batches.
        Returns a mapping from source term to translated term.
        Uses a persistent cache (a JSON file in the project-level cache folder)
        to avoid redundant API calls unless force_refresh is True.
        """
        if not force_refresh and self.translations_cache_fp.exists():
            cached_translations = FileManager.load_json(self.translations_cache_fp)
            logger.info(f"Loaded {len(cached_translations)} cached translations.")
        else:
            cached_translations = {}
            logger.info(
                "Translation cache not found or force_refresh enabled; starting with an empty cache."
            )

        # Determine which terms need translation.
        terms_to_translate = [term for term in terms if term not in cached_translations]

        if terms_to_translate:
            async with Translator() as translator:
                batch_size = 50
                new_translations = {}
                for i in range(0, len(terms_to_translate), batch_size):
                    batch = terms_to_translate[i : i + batch_size]
                    batch_translations = await translator.translate(
                        batch, dest=self.lang_to, src=self.lang_from
                    )
                    if not isinstance(batch_translations, list):
                        batch_translations = [batch_translations]
                    for term, trans in zip(batch, batch_translations):
                        new_translations[term] = trans.text.lower()
                        logger.info(f"Translated '{term}' -> '{trans.text.lower()}'")
                # Update the cache with new translations and save it.
                cached_translations.update(new_translations)
                FileManager.save_json(self.translations_cache_fp, cached_translations)
                logger.info(f"Saved {len(new_translations)} new translations to cache.")

        return {term: cached_translations[term] for term in terms}

    @staticmethod
    def find_matches(terms: List[str], taxonomy_terms: Set[str]) -> Set[str]:
        """Find matching terms in the given taxonomy."""
        return {term for term in terms if term.lower() in taxonomy_terms}

    @timeit
    def add_new_terms(self, new_terms: List[Tuple[str, str]], github_username: str):
        """
        Update the taxonomy file with new term pairs (source, translated).
        If a matching block exists for the English term, the German term is appended.
        A new branch is created locally with the GitHub username as a prefix.
        No commit or push is performed.
        """
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split("\n\n")
        added_terms = []

        for source_term, translated_term in new_terms:
            found_block = False
            for i, block in enumerate(blocks):
                lines = block.split("\n")
                for line in lines:
                    if line.startswith(f"{self.lang_to}:"):
                        english_terms = (
                            line[len(f"{self.lang_to}:") :].strip().split(", ")
                        )
                        if translated_term in english_terms:
                            found_block = True
                            # Look for an existing German line.
                            de_line_index = -1
                            for j, block_line in enumerate(lines):
                                if block_line.startswith(f"{self.lang_from}:"):
                                    de_line_index = j
                                    break
                            if de_line_index != -1:
                                existing_terms = (
                                    lines[de_line_index][len(f"{self.lang_from}:") :]
                                    .strip()
                                    .split(", ")
                                )
                                if source_term not in existing_terms:
                                    lines[de_line_index] = (
                                        lines[de_line_index].rstrip()
                                        + f", {source_term}"
                                    )
                            else:
                                new_de_line = f"{self.lang_from}: {source_term}"
                                insert_index = 0
                                for j, block_line in enumerate(lines):
                                    if block_line.startswith(f"{self.lang_to}:"):
                                        continue
                                    if block_line > new_de_line:
                                        insert_index = j
                                        break
                                    insert_index = j + 1
                                lines.insert(insert_index, new_de_line)
                            blocks[i] = "\n".join(lines)
                            added_terms.append((source_term, translated_term))
                            break
                if found_block:
                    break
            if not found_block:
                logger.warning(
                    f"No block found for English term: {translated_term}. Skipping adding German term: {source_term}."
                )

        updated_content = "\n\n".join(blocks)
        with open(self.taxonomy_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        logger.info("Taxonomy file updated.")

        # Create a new branch with the GitHub username as a prefix.
        branch_name = f"{github_username}_taxonomy_update_{self.lang_from}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating new branch '{branch_name}' for taxonomy update.")
        self.repo.git.checkout("-b", branch_name)
        logger.info(
            f"Branch '{branch_name}' created. Please review changes and commit manually."
        )

    @timeit
    async def process_new_ingredients(
        self, new_ingredients: List[str], force_refresh: bool = False
    ) -> Dict[str, str]:
        """
        Process new ingredient terms:
         1. Identify new terms from the provided list that are not already in the German taxonomy.
         2. Translate them (using force_refresh if requested).
         3. Return a mapping of new ingredient terms to their translations.
        Note: This method no longer updates the repository.
        """
        source_taxonomy = self.load_taxonomy(self.lang_from)
        target_taxonomy = self.load_taxonomy(self.lang_to)

        existing_terms = self.find_matches(new_ingredients, source_taxonomy)
        new_terms = [term for term in new_ingredients if term not in existing_terms]

        if not new_terms:
            return {}

        translations = await self.translate_terms(
            new_terms, force_refresh=force_refresh
        )
        valid_translations = {
            source_term: translated_term
            for source_term, translated_term in translations.items()
            if translated_term.lower() in target_taxonomy
        }
        return valid_translations


@timeit
async def get_new_ingredient_translations(
    new_ingredients: List[str], lang: Lang, force_refresh: bool = False
) -> Dict[str, str]:
    """
    Decoupled translation function:
    Process new ingredient terms and return a mapping of source term to translated term.
    This function does not update the taxonomy repository.
    """
    translator = IngredientTranslator(
        repo_path="/Users/maltewilhelm/Documents/projects/openfoodfacts-server",
        lang_from=lang.value,
        lang_to="en",
    )
    return await translator.process_new_ingredients(
        new_ingredients, force_refresh=force_refresh
    )


@timeit
def update_taxonomy_with_translations(
    translations: Dict[str, str], github_username: str = "github-throwaway"
):
    """
    Update the taxonomy file with new translations.
    This function assumes that the translations have already been processed.
    It handles repository update and taxonomy file modification.
    """
    translator = IngredientTranslator(
        repo_path="/Users/maltewilhelm/Documents/projects/openfoodfacts-server",
        lang_from="de",
        lang_to="en",
    )
    translator.update_repo()
    # Convert the translation mapping to a list of tuples.
    new_term_pairs = list(translations.items())
    if new_term_pairs:
        translator.add_new_terms(new_term_pairs, github_username)
        logger.info("Updated taxonomy with new term pairs:")
        for pair in new_term_pairs:
            logger.info(pair)
    else:
        logger.info("No new terms to add to the taxonomy.")


def get_new_ingredient_translations_sync(
    new_ingredients: List[str], lang: Lang, force_refresh: bool = False
) -> Dict[str, str]:
    """
    Synchronous wrapper for get_new_ingredient_translations.
    """
    return asyncio.run(
        get_new_ingredient_translations(
            new_ingredients, lang, force_refresh=force_refresh
        )
    )
