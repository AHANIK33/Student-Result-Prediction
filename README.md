# Student Result Prediction

Predicts whether a student passes or fails using demographic, lifestyle, and academic features, comparing an SVM (with hyperparameter tuning) against Logistic Regression.

## Overview

Using a dataset of 10,000 students, this project builds a preprocessing + classification pipeline to predict the binary `passed` outcome from a mix of demographic factors (age, gender, ethnicity, family income), lifestyle habits (sleep, stress, study hours), and academic scores (math, reading, writing, science, GPA).

## Dataset

- **File:** `students.csv`
- **Rows:** 10,000 students
- **Columns:** 25 (no missing values, duplicates removed)
- **Target:** `passed` (binary: 0/1)

| Category | Columns |
|---|---|
| Identifiers | `student_id` |
| Demographics | `age`, `gender`, `ethnicity` |
| Family/School | `parental_education`, `family_income`, `school_type`, `school_region` |
| Lifestyle | `study_hours_per_week`, `attendance_rate`, `extracurricular_activities`, `sports_participation`, `tutoring_sessions`, `parental_involvement`, `internet_access`, `has_laptop`, `sleep_hours`, `stress_level`, `motivation_score` |
| Academic scores | `reading_score`, `writing_score`, `math_score`, `science_score`, `overall_gpa` |
| Target | `passed` |

**Categorical value examples:**
- `ethnicity`: A, B, C, D, E
- `parental_education`: phd, master, bachelor, some_college, high_school, none
- `family_income`: low, middle, high
- `school_region`: urban, suburban, rural

## Approach

1. **EDA:** Histograms of numeric columns, correlation matrix + heatmap across academic/lifestyle scores
2. **Preprocessing** (via `ColumnTransformer`):
   - **Nominal columns** (`ethnicity`, `parental_education`, `family_income`, `school_region`, `school_type`, `gender`, `has_laptop`) → most-frequent imputation + one-hot encoding
   - **Numerical columns** (`math_score`, `reading_score`, `writing_score`, `sleep_hours`, `stress_level`, `motivation_score`, `science_score`, `overall_gpa`) → mean imputation + standard scaling
3. **Split:** 80% train / 20% test (`random_state=42`)
4. **Models trained:**
   - **SVC** with `GridSearchCV` (5-fold CV) tuning kernel (`linear`, `rbf`, `poly`), `C`, `gamma`, and `degree`
   - **Logistic Regression** (default settings)

## Results (Test Set)

| Model | Accuracy | R² | RMSE | MAE |
|---|---|---|---|---|
| SVM (best, grid search) | 99.70% | 0.814 | 0.087 | 0.0075 |
| Logistic Regression | 99.55% | 0.888 | 0.067 | 0.0045 |

Both models perform extremely well — likely because `overall_gpa` and the individual subject scores are highly predictive of `passed` on their own. The notebook favors the tuned SVM as the final model, though Logistic Regression is a simpler, nearly identical-performing alternative.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Place `students.csv` in your working directory (update the file path in the notebook if not using Colab).
2. Run `Student_result_prediction.ipynb` top to bottom.
3. Note: `GridSearchCV` over the full SVC parameter grid (linear/rbf/poly, multiple `C`/`gamma`/`degree` values, 5-fold CV) can take a while on 10,000 rows — expect longer runtimes than the other models in this repo.

## Project Structure

```
.
├── Student_result_prediction.ipynb   # EDA, preprocessing, training, evaluation
├── students.csv                       # Dataset (add your own)
└── README.md
```

## Future Improvements

- Since scores/GPA likely dominate the prediction, try a version excluding them to see how well demographic/lifestyle features alone predict passing
- Add feature importance analysis (e.g. permutation importance, SHAP) to interpret the SVM/Logistic Regression decisions
- Try Random Forest or Gradient Boosting for comparison
- Save the best pipeline with `joblib` for reuse without retraining
