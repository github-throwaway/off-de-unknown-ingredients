from pprint import pprint

from openfoodfacts import Lang

from ingredient_validator.ingredient_translator import (
    get_new_ingredient_translations_sync,
)
from ingredient_validator.ingredients_validator import IngredientValidator
from ingredient_validator.utils import (
    export_food_misspellings,
    extract_food_words,
    timeit,
)


@timeit
def main() -> None:
    validator = IngredientValidator(Lang.de)

    validator.fetch_unknown_ingredients(force_refresh=False)
    validator.preprocess_ingredients(force_refresh=False)
    validator.validate_ingredients_with_dwds(force_refresh=False)
    validator.store_original_invalid()

    validator.validate_spell_corrections(
        dictionary_path="./assets/deu_news_2024_1M/converted_dictionary.txt"
    )

    validator.fetch_dornseiff_bedeutungsgruppe(
        corpus="deu_news_2024", force_refresh=False
    )
    validator.fetch_openthesaurus_category("Gastronomie/Kulinarik", force_refresh=False)

    combined_data = validator.combine_opent_and_dornseiff()
    unknown_valid_ingredients = extract_food_words(combined_data)

    pprint(validator.get_product_count(unknown_valid_ingredients))

    export_food_misspellings(
        validator, unknown_valid_ingredients, "food_related_misspellings.csv"
    )

    get_new_ingredient_translations_sync(
        unknown_valid_ingredients, validator.lang, force_refresh=False
    )


if __name__ == "__main__":
    main()
