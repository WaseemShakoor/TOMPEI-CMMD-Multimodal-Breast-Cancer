# ============================================================
# TOMPEI-CMMD
# CREATE CLEAN BREAST-LEVEL CLINICAL DATASET
#
# Purpose:
#   - Use only breasts with both clinical + image data
#   - Exclude Exclusion and Invisible records
#   - Create binary target
#   - Standardise useful clinical variables
#   - Preserve BOTH image paths for every breast
#   - Save final modelling dataset
#
# Binary target:
#   Malignant      -> 1
#   Benign/Normal  -> 0
#
# IMPORTANT:
#   Raw data are NOT modified.
#   Missing clinical values are NOT imputed here.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

CLINICAL_FILE = (
    PROJECT_ROOT
    / "01_Data_Raw"
    / "01_Data_Raw"
    / "Tabular_Dataset"
    / "TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "02_Data_Cleaning"
    / "audit_outputs"
    / "breast_level_master_mapping_with_files.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_Data_Processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("TOMPEI-CMMD CLEAN BREAST-LEVEL DATASET CREATION")
print("=" * 75)

print("\nLoading raw clinical data...")

clinical = pd.read_excel(
    CLINICAL_FILE,
    sheet_name="Imaging Diagnosis Details Sheet"
)

print(f"Raw clinical records: {len(clinical):,}")


print("\nLoading breast-level image mapping...")

mapping = pd.read_csv(
    MAPPING_FILE
)

print(f"Mapping records: {len(mapping):,}")


# ============================================================
# 3. NORMALISE RAW CLINICAL IDENTIFIERS
# ============================================================

clinical["patient_id"] = (
    clinical["ID"]
    .astype("string")
    .str.strip()
)

clinical["breast_side"] = (
    clinical["LeftRight"]
    .astype("string")
    .str.strip()
    .str.upper()
)

clinical["classification"] = (
    clinical["classification"]
    .astype("string")
    .str.strip()
)

clinical["patient_breast_id"] = (
    clinical["patient_id"]
    + "_"
    + clinical["breast_side"]
)


# ============================================================
# 4. REMOVE ROW WITH INVALID/MISSING IDENTIFIER
# ============================================================

clinical = clinical[
    clinical["patient_id"].notna()
    &
    clinical["breast_side"].isin(["L", "R"])
].copy()


# ============================================================
# 5. STANDARDISE COLUMN NAMES
# ============================================================

clinical = clinical.rename(
    columns={
        "Age": "age",
        "Breast density": "breast_density",
        "BI-RADS\nCategory": "bi_rads",
        "Mass": "mass",
        "Calcification": "calcification",
        "Other findings": "other_findings"
    }
)


# ============================================================
# 6. SELECT USEFUL CLINICAL FIELDS
# ============================================================

clinical_columns = [
    "patient_breast_id",
    "patient_id",
    "breast_side",
    "classification",
    "age",
    "breast_density",
    "bi_rads",
    "mass",
    "calcification",
    "other_findings"
]

clinical_selected = clinical[
    clinical_columns
].copy()


# ============================================================
# 7. CHECK BREAST-LEVEL UNIQUENESS
# ============================================================

duplicates = clinical_selected[
    clinical_selected.duplicated(
        subset=["patient_breast_id"],
        keep=False
    )
]

print(
    "\nDuplicate patient-breast clinical records:",
    len(duplicates)
)

if len(duplicates) > 0:

    duplicates.to_csv(
        OUTPUT_DIR /
        "duplicate_patient_breast_records.csv",
        index=False
    )

    print(
        "WARNING: Duplicate breast-level records found."
    )


# ============================================================
# 8. KEEP ONLY MATCHED IMAGE BREASTS
# ============================================================

mapping_matched = mapping[
    mapping["match_status"] == "Matched"
].copy()

print(
    "\nMatched breast records before filtering:",
    len(mapping_matched)
)


# ============================================================
# 9. MERGE CLINICAL + IMAGE MAPPING
# ============================================================

merged = mapping_matched[
    [
        "patient_breast_id",
        "image_count",
        "image_files"
    ]
].merge(
    clinical_selected,
    on="patient_breast_id",
    how="left",
    validate="one_to_one"
)

print(
    "Merged matched breast records:",
    len(merged)
)


# ============================================================
# 10. CHECK MISSING CLASSIFICATION
# ============================================================

missing_class = (
    merged["classification"]
    .isna()
    .sum()
)

print(
    "Matched breasts with missing classification:",
    missing_class
)


# ============================================================
# 11. SHOW ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("ORIGINAL MATCHED CLASS DISTRIBUTION")
print("=" * 75)

print(
    merged["classification"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 12. EXCLUDE NON-MODELLING CATEGORIES
# ============================================================

excluded_labels = [
    "Exclusion"
]

excluded = merged[
    merged["classification"].isin(
        excluded_labels
    )
].copy()

excluded.to_csv(
    OUTPUT_DIR /
    "excluded_breast_records.csv",
    index=False
)

clean = merged[
    ~merged["classification"].isin(
        excluded_labels
    )
].copy()


# Remove missing classifications if any
clean = clean[
    clean["classification"].notna()
].copy()


# ============================================================
# 13. CREATE BINARY TARGET
# ============================================================

target_map = {
    "Normal": 0,
    "Benign": 0,
    "Malignant": 1,
    "Invisible": 1
}

clean["target"] = (
    clean["classification"]
    .map(target_map)
)


# Safety check
unexpected = clean[
    clean["target"].isna()
]

if len(unexpected) > 0:

    print(
        "\nERROR: Unexpected classification values found:"
    )

    print(
        unexpected[
            "classification"
        ]
        .value_counts()
    )

    raise ValueError(
        "Unexpected classification values detected."
    )

clean["target"] = (
    clean["target"]
    .astype(int)
)


# ============================================================
# 14. CLEAN AGE
# ============================================================

clean["age"] = pd.to_numeric(
    clean["age"],
    errors="coerce"
)

# IMPORTANT:
# Do NOT impute age here.
# Imputation will be performed later using TRAINING data only.


# ============================================================
# 15. CLEAN BI-RADS
# ============================================================

clean["bi_rads"] = pd.to_numeric(
    clean["bi_rads"],
    errors="coerce"
)


# ============================================================
# 16. STANDARDISE BREAST DENSITY TEXT
# ============================================================

clean["breast_density"] = (
    clean["breast_density"]
    .astype("string")
    .str.strip()
)

clean.loc[
    clean["breast_density"].isin(
        ["", "nan", "None"]
    ),
    "breast_density"
] = pd.NA


# ============================================================
# 17. CLEAN TEXT FEATURES
# ============================================================

text_columns = [
    "mass",
    "calcification",
    "other_findings"
]

for column in text_columns:

    clean[column] = (
        clean[column]
        .astype("string")
        .str.strip()
    )

    clean.loc[
        clean[column].isin(
            ["", "nan", "None"]
        ),
        column
    ] = pd.NA


# ============================================================
# 18. SPLIT TWO IMAGE PATHS
# ============================================================

image_paths = (
    clean["image_files"]
    .astype(str)
    .str.split(
        " | ",
        regex=False
    )
)


clean["number_of_image_paths"] = (
    image_paths.apply(len)
)

print("\nImages per final breast:")

print(
    clean[
        "number_of_image_paths"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 19. REQUIRE EXACTLY TWO IMAGES PER BREAST
# ============================================================

invalid_image_counts = clean[
    clean["number_of_image_paths"] != 2
].copy()

if len(invalid_image_counts) > 0:

    invalid_image_counts.to_csv(
        OUTPUT_DIR /
        "invalid_image_count_breasts.csv",
        index=False
    )

    raise ValueError(
        "Some breasts do not contain exactly two images."
    )


clean["image_file_1"] = (
    image_paths.apply(
        lambda x: x[0]
    )
)

clean["image_file_2"] = (
    image_paths.apply(
        lambda x: x[1]
    )
)


# ============================================================
# 20. ADD DATASET SOURCE
# ============================================================

clean["dataset"] = (
    clean["patient_id"]
    .astype(str)
    .str[:2]
)


# ============================================================
# 21. CHECK FINAL UNIQUENESS
# ============================================================

duplicate_final = clean.duplicated(
    subset=["patient_breast_id"]
).sum()

print(
    "\nDuplicate final breast IDs:",
    duplicate_final
)

if duplicate_final != 0:

    raise ValueError(
        "Final dataset contains duplicate breast IDs."
    )


# ============================================================
# 22. REORDER FINAL COLUMNS
# ============================================================

final_columns = [
    "patient_breast_id",
    "patient_id",
    "breast_side",
    "dataset",
    "classification",
    "target",

    "age",
    "breast_density",
    "bi_rads",
    "mass",
    "calcification",
    "other_findings",

    "image_count",
    "image_file_1",
    "image_file_2"
]

clean = clean[
    final_columns
].copy()


# ============================================================
# 23. FINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("FINAL BINARY CLASS DISTRIBUTION")
print("=" * 75)

class_summary = (
    clean["target"]
    .value_counts()
    .sort_index()
    .rename_axis("target")
    .reset_index(name="breast_count")
)

class_summary["class_name"] = (
    class_summary["target"]
    .map({
        0: "Non-malignant",
        1: "Malignant"
    })
)

class_summary["percentage"] = (
    class_summary["breast_count"]
    / len(clean)
    * 100
).round(2)

print(
    class_summary[
        [
            "target",
            "class_name",
            "breast_count",
            "percentage"
        ]
    ].to_string(index=False)
)


# ============================================================
# 24. FINAL DATA QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLEAN DATASET SUMMARY")
print("=" * 75)

print(
    f"\nFinal breast records: {len(clean):,}"
)

print(
    f"Unique patients: "
    f"{clean['patient_id'].nunique():,}"
)

print(
    f"Excluded records: {len(excluded):,}"
)

print(
    f"Missing age: "
    f"{clean['age'].isna().sum():,}"
)

print(
    f"Missing breast density: "
    f"{clean['breast_density'].isna().sum():,}"
)

print(
    f"Missing BI-RADS: "
    f"{clean['bi_rads'].isna().sum():,}"
)

print(
    f"Missing mass description: "
    f"{clean['mass'].isna().sum():,}"
)

print(
    f"Missing calcification description: "
    f"{clean['calcification'].isna().sum():,}"
)

print(
    f"Missing other findings: "
    f"{clean['other_findings'].isna().sum():,}"
)


# ============================================================
# 25. SAVE FINAL CLEAN DATASET
# ============================================================

FINAL_FILE = (
    OUTPUT_DIR /
    "tompei_cmmd_clean_breast_level_dataset.csv"
)

clean.to_csv(
    FINAL_FILE,
    index=False
)


# ============================================================
# 26. SAVE CLASS SUMMARY
# ============================================================

class_summary.to_csv(
    OUTPUT_DIR /
    "final_binary_class_distribution.csv",
    index=False
)


# ============================================================
# 27. SAVE MISSING VALUE SUMMARY
# ============================================================

missing_summary = pd.DataFrame({
    "column": clean.columns,
    "missing_count": [
        clean[col].isna().sum()
        for col in clean.columns
    ],
    "missing_percentage": [
        round(
            clean[col].isna().mean() * 100,
            2
        )
        for col in clean.columns
    ]
})

missing_summary.to_csv(
    OUTPUT_DIR /
    "final_clean_dataset_missing_values.csv",
    index=False
)


# ============================================================
# 28. SAVE CLEANING SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR /
    "clinical_cleaning_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TOMPEI-CMMD CLINICAL CLEANING SUMMARY\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Matched breasts before exclusions: "
        f"{len(merged):,}\n"
    )

    f.write(
        f"Excluded Exclusion/Invisible records: "
        f"{len(excluded):,}\n"
    )

    f.write(
        f"Final modelling breasts: "
        f"{len(clean):,}\n"
    )

    f.write(
        f"Unique patients represented: "
        f"{clean['patient_id'].nunique():,}\n"
    )

    f.write(
        "\nTarget definition:\n"
    )

    f.write(
        "0 = Non-malignant (Normal + Benign)\n"
    )

    f.write(
        "1 = Malignant\n"
    )

    f.write(
    "Excluded labels:\n"
    )

    f.write(
    "Exclusion\n"
    )

    f.write(
    "\nInvisible cases were retained as malignant "
    "because the TOMPEI-CMMD documentation defines "
    "them as malignant breasts without an identifiable "
    "lesion location on imaging.\n"
    )

    f.write(
        "\nMissing values were preserved and "
        "NOT imputed during cleaning.\n"
    )

    f.write(
        "Imputation will be performed after "
        "train/validation/test splitting to "
        "prevent data leakage.\n"
    )


# ============================================================
# 29. FINISH
# ============================================================

print("\n" + "=" * 75)
print("CLINICAL DATA CLEANING COMPLETE")
print("=" * 75)

print("\nGenerated files:")

print(
    "- tompei_cmmd_clean_breast_level_dataset.csv"
)

print(
    "- final_binary_class_distribution.csv"
)

print(
    "- final_clean_dataset_missing_values.csv"
)

print(
    "- excluded_breast_records.csv"
)

print(
    "- clinical_cleaning_summary.txt"
)

print(
    "\nRaw TOMPEI-CMMD files were NOT modified."
)

print("\nNEXT STEP:")

print(
    "Inspect the final cleaned dataset and create "
    "patient-level train/validation/test splits."
)