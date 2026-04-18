import json, re, pandas as pd
from pathlib import Path

# Source CSV names can be adjusted
review_files = [
    'reviews_2021.04.01-2022.03.31.csv',
    'reviews_2022.04.01-2023.03.31.csv',
    'reviews_2023.04.01-2024.03.31.csv',
    'reviews_2024.04.01-2025.03.31.csv',
    'reviews_2025.04.01-2026.03.31.csv',
]

# This helper script rebuilds review_summary.json from raw CSV files.
print('Use the notebook/build pipeline version from ChatGPT session for richer extraction.')
