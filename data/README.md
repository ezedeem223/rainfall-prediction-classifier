# Data Directory

This project uses the Australian weather classification dataset commonly shared on Kaggle:

- Source: `https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package`
- Expected local file: `data/raw/weatherAUS.csv`
- Target column for the refactored package: `RainTomorrow`

## Expected layout

```text
data/
├── README.md
└── raw/
    └── weatherAUS.csv
```

## How to obtain the data

1. Download the dataset from Kaggle.
2. Extract the CSV locally.
3. Place the file at `data/raw/weatherAUS.csv`.

The dataset is not committed to this repository. The `.gitignore` excludes `data/raw/` so large raw files do not end up in version control.

## Notes on preprocessing

- The refactored training flow assumes the dataset includes the standard weather columns, including `Date`, `Location`, `RainToday`, and `RainTomorrow`.
- By default the scripts drop rows with missing values, mirroring the preserved notebook's simple preprocessing strategy.
- The package also derives a `Season` feature from `Date` and drops the original date column before modeling.
- The preserved historical notebook in `notebooks/rainfall_prediction_classifier.ipynb` currently uses an IBM-hosted mirror (`weatherAUS-2.csv`), narrows the analysis to Melbourne-area locations, and temporarily renames the target for a same-day rainfall variant. That notebook is kept as historical reference rather than the canonical training entrypoint.
