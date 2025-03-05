import csv
import functools
import re
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List

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


def convert_dictionary(self, input_path: Path, output_path: Path) -> bool:
    """Convert dictionary file into correct format for SymSpell."""
    print("Converting dictionary...")
    try:
        with (
            input_path.open("r", encoding="utf-8") as infile,
            output_path.open("w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                parts = line.strip().split()
                if len(parts) >= 3:  # Ensure line has Word_ID, Word, Frequency
                    word_id, *word_parts, frequency = parts
                    word = " ".join(word_parts)

                    if word_id.isdigit() and int(word_id) > 100:
                        try:
                            frequency_int = int(frequency)
                            if self.is_german_word(word):
                                outfile.write(f"{word} {frequency_int}\n")
                        except ValueError:
                            continue
        print(f"Dictionary converted and saved to '{output_path}'.")
        return True
    except Exception as e:
        print(f"Error converting dictionary: {e}")
        return False


@timeit
def export_food_misspellings(
    validator, unknown_valid_ingredients, output_csv: str
) -> None:
    """
    Export food-related misspellings to a CSV file with headings.

    This function obtains the food-related misspelling mapping from the validator,
    calculates the frequency counts, and writes the results to a CSV file.

    Parameters:
      - validator: an instance of IngredientValidator.
      - unknown_valid_ingredients: set of validated food words.
      - output_csv: Path for the output CSV file.
    """
    # Get mapping of misspelled words to their corrected versions.
    food_corrections = validator.get_food_related_misspelling_mapping(
        final_food_words=set(unknown_valid_ingredients)
    )

    # Get frequency counts for the misspelled ingredients.
    frequency_counts = validator.get_product_count(set(food_corrections.keys()))

    # Prepare the data and write to CSV.
    with open(Path(output_csv), mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row.
        writer.writerow(
            ["Misspelled", "Corrected", "Ingredient Tag", "Product Count", "URL"]
        )
        # Write each row.
        for misspelled, correction in food_corrections.items():
            ingredient_id = f"de:{misspelled.lower()}"
            count = frequency_counts.get(ingredient_id, 0)
            url = f"https://de.openfoodfacts.org/facets/zutaten/{misspelled.lower()}"
            writer.writerow([misspelled, correction, ingredient_id, count, url])

    logger.info("Data exported successfully to food_related_misspellings.csv")


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


def extract_food_words(leipzig: Dict[str, List[str]]) -> List[str]:
    """
    Given a Leipzig groups dictionary (word -> list of category strings),
    return a sorted list of words that are likely food-related.

    A word is considered food-related if any of its categories matches one of:
      - A food category code (16.5, 16.6, 16.7, 16.8, 16.13, 16.14, or 16.15).
      - Explicitly labeled as "Gastronomie/Kulinarik".
    """
    # Precompile the regex pattern for performance and clarity.
    pattern = re.compile(
        r"(16\.(?:5|6|7|8|13|14|15)|Gastronomie/Kulinarik|Kochkunst|Nahrungsmittel|Nahrung)"
    )
    food_words = sorted(
        word
        for word, categories in leipzig.items()
        if any(pattern.search(cat) for cat in categories)
    )
    logger.info(
        f"Found {len(food_words)} food related words that are currently unknown."
    )
    return food_words
