from enum import Enum
##
# When we have hard to classify scenes (with only common objects present, e.g. wall, floor, door),
# then we may want to use CVM only and not LLM or we might want to use both and compare or we may
# want to take into account LLM results too (failures really) to make a fair comparison to CVM.
##
class HardClassBehaviour(Enum):
    CLASSIFY_WITH_CVM_ONLY = 1 # Use only CVM result for hard to classify cases
    CLASSIFY_WITH_LLM_ONLY = 2 # Use only LLM result for hard to classify cases
    CLASSIFY_WITH_CVM_AND_LLM = 3 # Use both CVM and LLM results for hard to classify cases (for fair comparison)
    CLASSIFY_WITH_NEITHER = 4 # Skip hard to classify cases altogether (for fair comparison)
    CLASSIFY_HARD_CASES_ONLY_CVM = 5 # Only classify hard scenes and only use CVM
    GUESS_HARD_CASES_ONLY_CVM = 6 # Try guessing (random selection) for hard cases
    CLASSIFY_WITH_CVM_WHAT_LLM_CANNOT = 7 # Use CVM result for the cases where LLM result is NOT_KNOWN, but ignore the hard to classify cases where LLM didn't classify (NOT_CLASSIFIED)
    GUESS_HARD_CASES_WHEN_LLM = 8 # Try guessing (random selection) for hard cases when we are classifying with LLM
