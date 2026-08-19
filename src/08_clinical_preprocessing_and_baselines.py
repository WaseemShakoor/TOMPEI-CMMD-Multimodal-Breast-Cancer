# ============================================================
# TOMPEI-CMMD
# CLINICAL PREPROCESSING + BASELINE MODELS
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. HistGradientBoosting
#
# Feature Sets:
#   A = age + breast_density + breast_side
#   B = age + breast_density + breast_side + bi_rads
#
# Method:
#   - Fit preprocessing on TRAIN only
#   - Use VALIDATION for model selection
#   - Evaluate final selected model on TEST once
#
# Outputs:
#   - validation metrics
#   - final test metrics
#   - predictions
#   - trained model
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_Results"
    / "clinical_baselines"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "04_Models"
    / "saved_models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. LOAD SPLITS
# ============================================================

train_df = pd.read_csv(
    SPLIT_DIR / "train_breast_level.csv"
)

val_df = pd.read_csv(
    SPLIT_DIR / "validation_breast_level.csv"
)

test_df = pd.read_csv(
    SPLIT_DIR / "test_breast_level.csv"
)

print("=" * 75)
print("CLINICAL BASELINE MODELLING")
print("=" * 75)

print(
    f"\nTrain: {len(train_df):,} breasts"
)

print(
    f"Validation: {len(val_df):,} breasts"
)

print(
    f"Test: {len(test_df):,} breasts"
)

# ============================================================
# 3. FEATURE SETS
# ============================================================

FEATURE_SETS = {

    "Set_A_without_BIRADS": {
        "numeric": ["age"],
        "categorical": [
            "breast_density",
            "breast_side"
        ]
    },

    "Set_B_with_BIRADS": {
        "numeric": [
            "age",
            "bi_rads"
        ],
        "categorical": [
            "breast_density",
            "breast_side"
        ]
    }
}

TARGET = "target"

# ============================================================
# 4. PREPROCESSING BUILDER
# ============================================================

def build_preprocessor(
    numeric_features,
    categorical_features
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor

# ============================================================
# 5. MODELS
# ============================================================

MODELS = {

    "LogisticRegression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

    "HistGradientBoosting":
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            random_state=42
        )
}

# ============================================================
# 6. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics = {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall_sensitivity":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob
            ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return metrics

# ============================================================
# 7. VALIDATION EXPERIMENTS
# ============================================================

validation_results = []

trained_pipelines = {}

for feature_set_name, feature_info in FEATURE_SETS.items():

    numeric_features = (
        feature_info["numeric"]
    )

    categorical_features = (
        feature_info["categorical"]
    )

    all_features = (
        numeric_features
        + categorical_features
    )

    X_train = train_df[
        all_features
    ].copy()

    y_train = train_df[
        TARGET
    ].copy()

    X_val = val_df[
        all_features
    ].copy()

    y_val = val_df[
        TARGET
    ].copy()

    print("\n" + "=" * 75)
    print(
        f"FEATURE SET: {feature_set_name}"
    )
    print("=" * 75)

    print(
        "Features:",
        all_features
    )

    for model_name, model in MODELS.items():

        print(
            f"\nTraining: "
            f"{model_name}"
        )

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        val_pred = pipeline.predict(
            X_val
        )

        # Probability
        if hasattr(
            pipeline,
            "predict_proba"
        ):

            val_prob = (
                pipeline.predict_proba(
                    X_val
                )[:, 1]
            )

        else:

            val_prob = (
                pipeline.decision_function(
                    X_val
                )
            )

        metrics = calculate_metrics(
            y_val,
            val_pred,
            val_prob
        )

        result = {
            "feature_set":
                feature_set_name,
            "model":
                model_name,
            **metrics
        }

        validation_results.append(
            result
        )

        trained_pipelines[
            (
                feature_set_name,
                model_name
            )
        ] = pipeline

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall_sensitivity']:.4f}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.4f}"
        )

# ============================================================
# 8. SAVE VALIDATION RESULTS
# ============================================================

validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = (
    validation_results_df
    .sort_values(
        by=[
            "roc_auc",
            "f1"
        ],
        ascending=False
    )
)

validation_results_df.to_csv(
    OUTPUT_DIR /
    "clinical_validation_results.csv",
    index=False
)

print("\n" + "=" * 75)
print("VALIDATION MODEL RANKING")
print("=" * 75)

print(
    validation_results_df[
        [
            "feature_set",
            "model",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc"
        ]
    ].to_string(
        index=False
    )
)

# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_row = (
    validation_results_df
    .iloc[0]
)

best_feature_set = (
    best_row[
        "feature_set"
    ]
)

best_model_name = (
    best_row[
        "model"
    ]
)

print("\n" + "=" * 75)
print("SELECTED MODEL")
print("=" * 75)

print(
    "Feature set:",
    best_feature_set
)

print(
    "Model:",
    best_model_name
)

print(
    "Validation ROC-AUC:",
    round(
        best_row[
            "roc_auc"
        ],
        4
    )
)

# ============================================================
# 10. PREPARE FINAL TRAINING DATA
# ============================================================

best_features_info = (
    FEATURE_SETS[
        best_feature_set
    ]
)

best_numeric = (
    best_features_info[
        "numeric"
    ]
)

best_categorical = (
    best_features_info[
        "categorical"
    ]
)

best_features = (
    best_numeric
    + best_categorical
)

# Combine TRAIN + VALIDATION
final_train_df = pd.concat(
    [
        train_df,
        val_df
    ],
    ignore_index=True
)

X_final_train = (
    final_train_df[
        best_features
    ]
)

y_final_train = (
    final_train_df[
        TARGET
    ]
)

X_test = (
    test_df[
        best_features
    ]
)

y_test = (
    test_df[
        TARGET
    ]
)

# ============================================================
# 11. REBUILD BEST MODEL
# ============================================================

# fresh copy of estimator
if best_model_name == "LogisticRegression":

    best_estimator = (
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )

elif best_model_name == "RandomForest":

    best_estimator = (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    )

elif best_model_name == "HistGradientBoosting":

    best_estimator = (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            random_state=42
        )
    )

else:

    raise ValueError(
        "Unknown selected model."
    )

final_preprocessor = (
    build_preprocessor(
        best_numeric,
        best_categorical
    )
)

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            final_preprocessor
        ),
        (
            "model",
            best_estimator
        )
    ]
)

# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

print("\nTraining final selected model...")
print(
    "Using Train + Validation data."
)

final_pipeline.fit(
    X_final_train,
    y_final_train
)

# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

test_pred = (
    final_pipeline.predict(
        X_test
    )
)

test_prob = (
    final_pipeline.predict_proba(
        X_test
    )[:, 1]
)

test_metrics = (
    calculate_metrics(
        y_test,
        test_pred,
        test_prob
    )
)

print("\n" + "=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

for key, value in test_metrics.items():

    if isinstance(
        value,
        (float, np.floating)
    ):

        print(
            f"{key}: "
            f"{value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )

# ============================================================
# 14. SAVE FINAL TEST METRICS
# ============================================================

test_metrics_df = pd.DataFrame(
    [
        {
            "feature_set":
                best_feature_set,
            "model":
                best_model_name,
            **test_metrics
        }
    ]
)

test_metrics_df.to_csv(
    OUTPUT_DIR /
    "clinical_final_test_metrics.csv",
    index=False
)

# ============================================================
# 15. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = (
    test_df[
        [
            "patient_breast_id",
            "patient_id",
            "breast_side",
            "classification",
            "target"
        ]
    ]
    .copy()
)

test_predictions[
    "predicted_target"
] = test_pred

test_predictions[
    "malignant_probability"
] = test_prob

test_predictions.to_csv(
    OUTPUT_DIR /
    "clinical_test_predictions.csv",
    index=False
)

# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

MODEL_FILE = (
    MODEL_DIR /
    "best_clinical_model.joblib"
)

joblib.dump(
    final_pipeline,
    MODEL_FILE
)

# ============================================================
# 17. SAVE MODEL SELECTION SUMMARY
# ============================================================

selection_summary = pd.DataFrame(
    [
        {
            "selected_feature_set":
                best_feature_set,
            "selected_model":
                best_model_name,
            "validation_roc_auc":
                best_row[
                    "roc_auc"
                ],
            "validation_f1":
                best_row[
                    "f1"
                ],
            "test_roc_auc":
                test_metrics[
                    "roc_auc"
                ],
            "test_f1":
                test_metrics[
                    "f1"
                ],
            "test_accuracy":
                test_metrics[
                    "accuracy"
                ]
        }
    ]
)

selection_summary.to_csv(
    OUTPUT_DIR /
    "clinical_model_selection_summary.csv",
    index=False
)

# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 75)
print("CLINICAL BASELINE EXPERIMENT COMPLETE")
print("=" * 75)

print("\nSaved:")

print(
    "- clinical_validation_results.csv"
)

print(
    "- clinical_final_test_metrics.csv"
)

print(
    "- clinical_test_predictions.csv"
)

print(
    "- clinical_model_selection_summary.csv"
)

print(
    "- best_clinical_model.joblib"
)

print("\nNEXT STEP:")
print(
    "Review clinical results, then move to "
    "mammography image preprocessing."
)