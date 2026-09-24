# California House Price Prediction (Regression)

A machine learning project that predicts median house values for California districts from census data. It compares a linear model with a Random Forest, tests for overfitting, and uses cross-validation to check how far the results can be trusted.

**Tools:** Python, pandas, scikit-learn, Google Colab

## Dataset

`california_housing_train.csv` (17,000 districts, 1990 census), the sample file that ships with Google Colab. Each row is a district. Features include median income, housing age, rooms, bedrooms, population, households, latitude and longitude. The target is `median_house_value`. Prices are from 1990, so they do not reflect current values.

## Approach

1. **Baseline:** linear regression with 4 features
2. **Feature experiments:** removed and added features to see which ones matter
3. **Random Forest:** a non-linear model, compared on the same train/test split
4. **Overfitting check:** compared train and test scores, and tried restricting the trees
5. **Cross-validation:** 5-fold, with and without shuffling

## Results

All scores are R² and average error on held-out test data (80/20 split, `random_state=42`).

| Model | Test R² | Avg error |
|---|---|---|
| Linear, 3 features (no income) | 0.144 | $85,359 |
| Linear, 4 features | 0.546 | $59,193 |
| Linear, all 8 features | 0.664 | $49,983 |
| **Random Forest, all 8 features** | **0.823** | **$32,360** |

**Feature importance (Random Forest):** median income 51.5%, longitude 16.7%, latitude 15.5%. The other five features together contribute about 16%.

**Overfitting:** the forest scored 0.975 on training data against 0.823 on test data. Limiting tree depth and leaf size shrank the gap but lowered the test score (0.818 and 0.802), so the default settings were kept.

## Key finding: how the data was split changed the answer

| 5-fold cross-validation | Random Forest (mean R²) | Linear (mean R²) |
|---|---|---|
| Shuffled folds | 0.821 (folds: 0.811–0.831) | 0.637 |
| Unshuffled folds | 0.386 (one fold: −0.299) | 0.548 |

The data file is **sorted by longitude**. Without shuffling, each cross-validation fold is a geographic strip, so the model was tested on regions it had not seen in training. The forest, which relies heavily on latitude and longitude, dropped sharply, and one fold scored worse than simply predicting the average price.

This is consistent with the forest struggling on unseen regions. I did not test that directly, for example by holding out one region and checking its error separately.

**Takeaway:** the 0.82 score applies to predicting districts in areas the model has training examples for. It should not be assumed to hold for a new region.

## Limitations

- 1990 data, so it says nothing about current prices
- Only one dataset and one train/test split for the tuning experiments
- Latitude and longitude are correlated, so the importance credit is split between them
- Feature importance shows what the model used, not what causes prices



## Author

**[Your Name]** · [LinkedIn link] · [Email]
