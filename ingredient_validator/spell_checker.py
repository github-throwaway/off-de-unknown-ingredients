from pathlib import Path
from typing import Optional, List, Dict
from symspellpy.symspellpy import SymSpell, Verbosity  # Ensure symspellpy is installed
from ingredient_validator.utils import timeit


class SpellChecker:
    """Handles spell checking operations using SymSpell."""

    def __init__(self, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self.sym_spell: Optional[SymSpell] = None

    @timeit
    def load_dictionary(self, dictionary_path: Path) -> bool:
        """Load dictionary for spell checking."""
        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7
        )
        try:
            self.sym_spell.load_dictionary(
                str(dictionary_path), term_index=0, count_index=1
            )
            return True
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return False

    @timeit
    def get_corrections(self, words: List[str]) -> Dict[str, List[str]]:
        if not self.sym_spell:
            return {}

        corrections = {}
        for word in words:
            suggestions = self.sym_spell.lookup(
                word, Verbosity.TOP, max_edit_distance=self.max_edit_distance
            )
            filtered_suggestions = [sug.term for sug in suggestions]
            if filtered_suggestions:
                corrections[word] = filtered_suggestions
        return corrections


async def check_spelling(
    words: List[str], dictionary_path: Path
) -> Dict[str, List[str]]:
    """
    Asynchronously check the spelling of given words using the provided dictionary.
    Returns a mapping from misspelled word to a list of correction suggestions.
    """
    checker = SpellChecker()
    if not checker.load_dictionary(dictionary_path):
        return {}
    return checker.get_corrections(words)
