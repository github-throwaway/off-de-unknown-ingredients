from pprint import pprint

from openfoodfacts import Lang

from ingredient_validator.ingredients_validator import IngredientValidator
from ingredient_validator.utils import extract_food_words
from ingredient_validator.utils import timeit


@timeit
def main() -> None:
    fetcher = IngredientValidator(Lang.de)

    # Step 1: Fetch raw unknown ingredients data
    fetcher.fetch_unknown_ingredients(force_refresh=False)

    # Step 2: Pre-process the data
    fetcher.preprocess_ingredients(force_refresh=False)

    # Step 3: Validate ingredients using DWDS API
    fetcher.validate_ingredients_with_dwds(force_refresh=False)

    # Step 4: Get Dornseiff-Bedeutungsgruppen
    fetcher.fetch_dornseiff_bedeutungsgruppe(
        corpus="deu_news_2024", force_refresh=False
    )

    # Step 5: OpenThesaurus
    fetcher.fetch_openthesaurus_category("Gastronomie/Kulinarik", force_refresh=False)
    res = fetcher.combine()

    unknown_valid_ingredients = extract_food_words(res)
    count = fetcher.get_product_count(unknown_valid_ingredients)
    pprint(count)


if __name__ == "__main__":
    main()
