"""
This file holds constants that will be used throughout the application.
"""

import numpy as np

ENCODING_UTF_8 = "utf-8"

# sklearn things
SEED_CONSTANT = 123

# File modes
FILE_MODE_READ = "r"
FILE_MODE_WRITE = "w"

# Current year for data filtering
LATEST_YEAR = 2019

# Labels for ML models
LABELS = np.array(["for respondent", "for petitioner"])
