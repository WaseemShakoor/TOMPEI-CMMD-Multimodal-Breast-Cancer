# ============================================================
# TOMPEI-CMMD
# CLINICAL BASELINE VISUALISATIONS
#
# Creates:
#   1. Validation ROC-AUC comparison
#   2. Validation metric comparison
#   3. BI-RADS ablation comparison
#   4. Final test confusion matrix
#   5. Final test ROC curve
#   6. Final test Precision-Recall curve
#   7. Logistic Regression coefficient plot
#
# IMPORTANT:
#   Models are NOT retrained.
#   Existing saved results are used.
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "05_Results"
    / "clinical_baselines"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "04_Models"
    / "saved_models"
    / "best_clinical_model.joblib"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD SAVED RESULTS
# ============================================================

validation_results = pd.read_csv(
    RESULTS_DIR
    / "clinical_validation_results.csv"
)

test_metrics = pd.read_csv(
    RESULTS_DIR
    / "clinical_final_test_metrics.csv"
)

test_predictions = pd.read_csv(
    RESULTS_DIR
    / "clinical_test_predictions.csv"
)


print("=" * 75)
print("CLINICAL BASELINE VISUALISATIONS")
print("=" * 75)

print("\nValidation experiments:",
      len(validation_results))

print("Test predictions:",
      len(test_predictions))


# ============================================================
# 3. FRIENDLY LABELS
# ============================================================

feature_labels = {
    "Set_A_without_BIRADS":
        "Without BI-RADS",

    "Set_B_with_BIRADS":
        "With BI-RADS"
}


validation_results[
    "feature_label"
] = validation_results[
    "feature_set"
].map(feature_labels)


validation_results[
    "experiment"
] = (
    validation_results["model"]
    + "\n"
    + validation_results["feature_label"]
)


# ============================================================
# 4. VALIDATION ROC-AUC COMPARISON
# ============================================================

roc_sorted = (
    validation_results
    .sort_values(
        "roc_auc",
        ascending=False
    )
)

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    roc_sorted["experiment"],
    roc_sorted["roc_auc"]
)

plt.ylabel(
    "Validation ROC-AUC"
)

plt.xlabel(
    "Model / Feature Set"
)

plt.title(
    "Clinical Model Validation ROC-AUC Comparison"
)

plt.ylim(
    0.50,
    1.00
)

plt.xticks(
    rotation=35,
    ha="right"
)

for bar, value in zip(
    bars,
    roc_sorted["roc_auc"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.006,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "01_validation_roc_auc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. VALIDATION METRIC COMPARISON
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1"
]

metric_names = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1"
]

plot_df = (
    validation_results[
        [
            "experiment"
        ]
        + metrics
    ]
    .set_index(
        "experiment"
    )
)

plot_df.columns = metric_names


ax = plot_df.plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title(
    "Validation Performance Across Clinical Models"
)

ax.set_ylabel(
    "Score"
)

ax.set_xlabel(
    "Model / Feature Set"
)

ax.set_ylim(
    0,
    1.05
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.legend(
    title="Metric",
    bbox_to_anchor=(
        1.02,
        1
    ),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "02_validation_metric_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. BI-RADS ABLATION COMPARISON
# ============================================================

ablation = (
    validation_results
    .pivot(
        index="model",
        columns="feature_label",
        values="roc_auc"
    )
)

ax = ablation.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_title(
    "Effect of BI-RADS on Validation ROC-AUC"
)

ax.set_ylabel(
    "ROC-AUC"
)

ax.set_xlabel(
    "Clinical Model"
)

ax.set_ylim(
    0.50,
    1.00
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Feature Set"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "03_birads_ablation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FINAL TEST CONFUSION MATRIX
# ============================================================

row = test_metrics.iloc[0]

tn = int(row["TN"])
fp = int(row["FP"])
fn = int(row["FN"])
tp = int(row["TP"])

confusion = np.array(
    [
        [tn, fp],
        [fn, tp]
    ]
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

image = ax.imshow(
    confusion
)

ax.set_title(
    "Final Clinical Model Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)

ax.set_xticks(
    [0, 1]
)

ax.set_yticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "Non-malignant",
        "Malignant"
    ]
)

ax.set_yticklabels(
    [
        "Non-malignant",
        "Malignant"
    ]
)

for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            confusion[i, j],
            ha="center",
            va="center",
            fontsize=14
        )

fig.colorbar(
    image,
    ax=ax
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "04_final_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. FINAL TEST ROC CURVE
# ============================================================

y_true = (
    test_predictions[
        "target"
    ].values
)

y_prob = (
    test_predictions[
        "malignant_probability"
    ].values
)


fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

roc_auc_value = auc(
    fpr,
    tpr
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=(
        f"Clinical model "
        f"(AUC = {roc_auc_value:.3f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate (Sensitivity)"
)

plt.title(
    "Final Clinical Model ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "05_final_test_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. FINAL TEST PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds_pr = (
    precision_recall_curve(
        y_true,
        y_prob
    )
)

pr_auc_value = (
    average_precision_score(
        y_true,
        y_prob
    )
)

baseline = (
    y_true.mean()
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=(
        f"Clinical model "
        f"(AP = {pr_auc_value:.3f})"
    )
)

plt.axhline(
    y=baseline,
    linestyle="--",
    label=(
        f"Class prevalence "
        f"({baseline:.3f})"
    )
)

plt.xlabel(
    "Recall (Sensitivity)"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final Clinical Model Precision-Recall Curve"
)

plt.legend(
    loc="lower left"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "06_final_test_precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print(
    "\nLoading final clinical model..."
)

pipeline = joblib.load(
    MODEL_FILE
)

preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)

model = (
    pipeline.named_steps[
        "model"
    ]
)


if hasattr(
    model,
    "coef_"
):

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = (
            model.coef_[0]
        )

        coefficient_df = pd.DataFrame({
            "feature":
                feature_names,
            "coefficient":
                coefficients
        })

        coefficient_df[
            "absolute_coefficient"
        ] = (
            coefficient_df[
                "coefficient"
            ].abs()
        )

        coefficient_df = (
            coefficient_df
            .sort_values(
                "absolute_coefficient",
                ascending=True
            )
        )

        coefficient_df.to_csv(
            RESULTS_DIR
            / "logistic_regression_coefficients.csv",
            index=False
        )

        plt.figure(
            figsize=(9, 7)
        )

        plt.barh(
            coefficient_df[
                "feature"
            ],
            coefficient_df[
                "coefficient"
            ]
        )

        plt.axvline(
            x=0,
            linewidth=1
        )

        plt.xlabel(
            "Logistic Regression Coefficient"
        )

        plt.ylabel(
            "Clinical Feature"
        )

        plt.title(
            "Final Clinical Model Feature Coefficients"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / "07_logistic_regression_coefficients.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Coefficient plot created successfully."
        )

    except Exception as error:

        print(
            "\nCould not create coefficient plot:"
        )

        print(error)

else:

    print(
        "\nFinal model is not a linear model."
    )

    print(
        "Coefficient plot skipped."
    )


# ============================================================
# 11. SAVE FIGURE INDEX
# ============================================================

figure_index = pd.DataFrame({

    "figure_number": [
        1, 2, 3, 4, 5, 6, 7
    ],

    "figure": [
        "Validation ROC-AUC Comparison",
        "Validation Metric Comparison",
        "BI-RADS Ablation Comparison",
        "Final Test Confusion Matrix",
        "Final Test ROC Curve",
        "Final Test Precision-Recall Curve",
        "Logistic Regression Coefficients"
    ],

    "filename": [
        "01_validation_roc_auc_comparison.png",
        "02_validation_metric_comparison.png",
        "03_birads_ablation_comparison.png",
        "04_final_test_confusion_matrix.png",
        "05_final_test_roc_curve.png",
        "06_final_test_precision_recall_curve.png",
        "07_logistic_regression_coefficients.png"
    ]
})

figure_index.to_csv(
    FIGURE_DIR
    / "clinical_figure_index.csv",
    index=False
)


# ============================================================
# 12. FINISH
# ============================================================

print("\n" + "=" * 75)
print("CLINICAL VISUALISATIONS COMPLETE")
print("=" * 75)

print("\nFigures saved to:")

print(
    FIGURE_DIR
)

print("\nGenerated:")

for figure in sorted(
    FIGURE_DIR.glob("*.png")
):

    print(
        "-",
        figure.name
    )

print(
    "\nNo models were retrained."
)

print(
    "\nNEXT STEP:"
)

print(
    "Begin mammography DICOM preprocessing "
    "for the image-based deep learning experiment."
)