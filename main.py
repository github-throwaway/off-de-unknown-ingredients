from openfoodfacts import Lang

from ingredient_validator.ingredients_validator import IngredientValidator
from ingredient_validator.utils import timeit


@timeit
def main():
    fetcher = IngredientValidator(Lang.de)

    # Step 1: Fetch raw unknown ingredients data
    fetcher.fetch_unknown_ingredients(force_refresh=False)

    # Step 2: Pre-process the data
    fetcher.preprocess_ingredients(force_refresh=False)


if __name__ == "__main__":
    main()
