# ============================================================
# TOMPEI-CMMD
# PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
#
# Purpose:
#   - Split by patient_id, NOT by breast
#   - Prevent patient-level data leakage
#   - Create fixed reproducible splits
#   - Preserve all breasts from the same patient in one split
#
# Split:
#   70% Train
#   15% Validation
#   15% Test
# ============================================================

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "tompei_cmmd_clean_breast_level_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# ============================================================
# 3. LOAD CLEAN DATASET
# ============================================================

print("=" * 75)
print("TOMPEI-CMMD PATIENT-LEVEL DATA SPLIT")
print("=" * 75)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal breast records: {len(df):,}")
print(f"Unique patients: {df['patient_id'].nunique():,}")

# ============================================================
# 4. CREATE PATIENT-LEVEL TARGET
# ============================================================

# For stratification, assign each patient a patient-level target.
# If either breast is malignant, the patient target becomes 1.

patient_df = (
    df.groupby("patient_id")
    .agg(
        patient_target=("target", "max"),
        breast_count=("patient_breast_id", "count")
    )
    .reset_index()
)

print(
    f"\nPatient-level records created: "
    f"{len(patient_df):,}"
)

print("\nPatient-level class distribution:")

print(
    patient_df["patient_target"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# 5. FIRST SPLIT:
#    TRAIN vs TEMP (VAL + TEST)
# ============================================================

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=patient_df["patient_target"]
)

# ============================================================
# 6. SECOND SPLIT:
#    VALIDATION vs TEST
# ============================================================

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_patients["patient_target"]
)

# ============================================================
# 7. PATIENT ID SETS
# ============================================================

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])

# ============================================================
# 8. CHECK FOR PATIENT OVERLAP
# ============================================================

train_val_overlap = train_ids.intersection(val_ids)
train_test_overlap = train_ids.intersection(test_ids)
val_test_overlap = val_ids.intersection(test_ids)

print("\n" + "=" * 75)
print("PATIENT OVERLAP CHECK")
print("=" * 75)

print(
    "Train / Validation overlap:",
    len(train_val_overlap)
)

print(
    "Train / Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation / Test overlap:",
    len(val_test_overlap)
)

if (
    len(train_val_overlap) > 0
    or len(train_test_overlap) > 0
    or len(val_test_overlap) > 0
):

    raise ValueError(
        "Patient overlap detected between splits."
    )

# ============================================================
# 9. ASSIGN BREAST RECORDS TO SPLITS
# ============================================================

train_df = df[
    df["patient_id"].isin(train_ids)
].copy()

val_df = df[
    df["patient_id"].isin(val_ids)
].copy()

test_df = df[
    df["patient_id"].isin(test_ids)
].copy()

# ============================================================
# 10. ADD SPLIT LABEL
# ============================================================

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# ============================================================
# 11. COMBINE MASTER SPLIT FILE
# ============================================================

full_split_df = pd.concat(
    [
        train_df,
        val_df,
        test_df
    ],
    ignore_index=True
)

# ============================================================
# 12. SPLIT SUMMARY FUNCTION
# ============================================================

def print_split_summary(name, split_df):

    print("\n" + "-" * 60)
    print(name.upper())
    print("-" * 60)

    print(
        f"Patients: "
        f"{split_df['patient_id'].nunique():,}"
    )

    print(
        f"Breasts: "
        f"{len(split_df):,}"
    )

    print("\nBreast-level target distribution:")

    counts = (
        split_df["target"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        split_df["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(summary.to_string())

# ============================================================
# 13. PRINT SPLIT SUMMARIES
# ============================================================

print("\n" + "=" * 75)
print("FINAL SPLIT SUMMARY")
print("=" * 75)

print_split_summary(
    "Train",
    train_df
)

print_split_summary(
    "Validation",
    val_df
)

print_split_summary(
    "Test",
    test_df
)

# ============================================================
# 14. SAVE BREAST-LEVEL SPLITS
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train_breast_level.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation_breast_level.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test_breast_level.csv",
    index=False
)

full_split_df.to_csv(
    OUTPUT_DIR / "all_breast_level_with_split.csv",
    index=False
)

# ============================================================
# 15. SAVE PATIENT-LEVEL SPLITS
# ============================================================

train_patients.to_csv(
    OUTPUT_DIR / "train_patients.csv",
    index=False
)

val_patients.to_csv(
    OUTPUT_DIR / "validation_patients.csv",
    index=False
)

test_patients.to_csv(
    OUTPUT_DIR / "test_patients.csv",
    index=False
)

# ============================================================
# 16. SAVE SUMMARY FILE
# ============================================================

summary_rows = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    for target_value in [0, 1]:

        count = (
            split_df["target"] == target_value
        ).sum()

        percentage = (
            count / len(split_df) * 100
        )

        summary_rows.append({
            "split": split_name,
            "target": target_value,
            "breast_count": count,
            "percentage": round(
                percentage,
                2
            ),
            "patient_count":
                split_df[
                    "patient_id"
                ].nunique()
        })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR /
    "split_distribution_summary.csv",
    index=False
)

# ============================================================
# 17. FINAL VALIDATION
# ============================================================

total_records = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

total_patients = (
    len(train_ids)
    + len(val_ids)
    + len(test_ids)
)

print("\n" + "=" * 75)
print("FINAL VALIDATION")
print("=" * 75)

print(
    f"\nOriginal breast records: "
    f"{len(df):,}"
)

print(
    f"Split breast records: "
    f"{total_records:,}"
)

print(
    f"Original patients: "
    f"{df['patient_id'].nunique():,}"
)

print(
    f"Split patients: "
    f"{total_patients:,}"
)

assert total_records == len(df)
assert total_patients == df["patient_id"].nunique()

print(
    "\nPatient-level split validation PASSED."
)

# ============================================================
# 18. FINISH
# ============================================================

print("\nGenerated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print("-", file.name)

print(
    "\nNEXT STEP:"
)

print(
    "Fit clinical preprocessing using TRAINING data only."
)