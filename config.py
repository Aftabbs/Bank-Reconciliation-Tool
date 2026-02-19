import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANK_STATEMENTS_PATH = os.path.join(PROJECT_ROOT, "bank_statements.csv")
CHECK_REGISTER_PATH = os.path.join(PROJECT_ROOT, "check_register.csv")
VALIDATED_PAIRS_PATH = os.path.join(PROJECT_ROOT, "validated_pairs.json")

AMOUNT_TOLERANCE = 0.01
DATE_LAG_FLAG_THRESHOLD_DAYS = 5
DATE_LAG_CONFIDENCE_PENALTY = 0.1

SVD_N_COMPONENTS = 30
ML_CONFIDENCE_THRESHOLD = 0.3

BANK_ID_PREFIX = "B"
REGISTER_ID_PREFIX = "R"
