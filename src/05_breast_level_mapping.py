# ============================================================
# TOMPEI-CMMD
# BREAST-LEVEL CLINICAL <-> IMAGE MAPPING
#
# Purpose:
#   Match clinical left/right breast records with DICOM images
#   using Patient ID + breast laterality.
#
# IMPORTANT:
#   - Raw data are NOT modified
#   - No labels are deleted
#   - No missing values are imputed
# ============================================================

from pathlib import Path
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

CLINICAL_FILE = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\01_Data_Raw\01_Data_Raw\Tabular_Dataset"
    r"\TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
)

DICOM_METADATA_FILE = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\02_Data_Cleaning\audit_outputs"
    r"\dicom_laterality_metadata_investigation.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\02_Data_Cleaning\audit_outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("TOMPEI-CMMD BREAST-LEVEL CLINICAL <-> IMAGE MAPPING")
print("=" * 75)

print("\nLoading clinical data...")

clinical = pd.read_excel(
    CLINICAL_FILE,
    sheet_name="Imaging Diagnosis Details Sheet"
)

print(
    f"Clinical records: {len(clinical):,}"
)

print("\nLoading DICOM metadata...")

images = pd.read_csv(
    DICOM_METADATA_FILE
)

print(
    f"DICOM image records: {len(images):,}"
)


# ============================================================
# 3. NORMALISE CLINICAL IDENTIFIERS
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


# ============================================================
# 4. NORMALISE IMAGE IDENTIFIERS
# ============================================================

images["patient_id"] = (
    images["PatientID"]
    .astype("string")
    .str.strip()
)

images["breast_side"] = (
    images["ImageLaterality"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ============================================================
# 5. KEEP VALID L/R VALUES
# ============================================================

clinical_valid = clinical[
    clinical["patient_id"].notna()
    &
    clinical["breast_side"].isin(["L", "R"])
].copy()

images_valid = images[
    images["patient_id"].notna()
    &
    images["breast_side"].isin(["L", "R"])
].copy()


# ============================================================
# 6. CREATE PATIENT-SIDE KEY
# ============================================================

clinical_valid["patient_breast_id"] = (
    clinical_valid["patient_id"]
    + "_"
    + clinical_valid["breast_side"]
)

images_valid["patient_breast_id"] = (
    images_valid["patient_id"]
    + "_"
    + images_valid["breast_side"]
)


# ============================================================
# 7. CLINICAL RECORD COUNT PER BREAST
# ============================================================

clinical_counts = (
    clinical_valid
    .groupby("patient_breast_id")
    .size()
    .rename("clinical_record_count")
)


# ============================================================
# 8. IMAGE COUNT PER BREAST
# ============================================================

image_counts = (
    images_valid
    .groupby("patient_breast_id")
    .size()
    .rename("image_count")
)


# ============================================================
# 9. CLASSIFICATION PER BREAST
# ============================================================

classification_by_breast = (
    clinical_valid
    .groupby(
        "patient_breast_id"
    )["classification"]
    .apply(
        lambda x:
        " + ".join(
            sorted(
                set(
                    x.dropna()
                    .astype(str)
                    .str.strip()
                )
            )
        )
    )
    .rename("classification")
)


# ============================================================
# 10. BASIC CLINICAL FEATURES
# ============================================================

def first_value(series):

    valid = series.dropna()

    if len(valid) == 0:
        return None

    return valid.iloc[0]


clinical_features = (
    clinical_valid
    .groupby("patient_breast_id")
    .agg({
        "patient_id": first_value,
        "breast_side": first_value,
        "Age": first_value,
        "Breast density": first_value,
        "BI-RADS\nCategory": first_value
    })
)


# ============================================================
# 11. BUILD MASTER BREAST TABLE
# ============================================================

all_breast_ids = sorted(
    set(clinical_counts.index)
    |
    set(image_counts.index)
)

master = pd.DataFrame(
    index=all_breast_ids
)

master.index.name = "patient_breast_id"

master = master.join(
    clinical_counts
)

master = master.join(
    image_counts
)

master = master.join(
    classification_by_breast
)

master = master.join(
    clinical_features
)


# ============================================================
# 12. FILL COUNTS
# ============================================================

master["clinical_record_count"] = (
    master["clinical_record_count"]
    .fillna(0)
    .astype(int)
)

master["image_count"] = (
    master["image_count"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 13. MATCH STATUS
# ============================================================

def match_status(row):

    clinical_exists = (
        row["clinical_record_count"] > 0
    )

    image_exists = (
        row["image_count"] > 0
    )

    if clinical_exists and image_exists:
        return "Matched"

    elif clinical_exists:
        return "Clinical only"

    elif image_exists:
        return "Image only"

    return "Unknown"


master["match_status"] = master.apply(
    match_status,
    axis=1
)


# ============================================================
# 14. RESET INDEX
# ============================================================

master = master.reset_index()


# ============================================================
# 15. SAVE MASTER TABLE
# ============================================================

MASTER_FILE = (
    OUTPUT_DIR /
    "breast_level_master_mapping.csv"
)

master.to_csv(
    MASTER_FILE,
    index=False
)


# ============================================================
# 16. SAVE MATCHED BREASTS
# ============================================================

matched = master[
    master["match_status"] == "Matched"
].copy()

MATCHED_FILE = (
    OUTPUT_DIR /
    "matched_breast_records.csv"
)

matched.to_csv(
    MATCHED_FILE,
    index=False
)


# ============================================================
# 17. SAVE UNMATCHED BREASTS
# ============================================================

clinical_only = master[
    master["match_status"] == "Clinical only"
].copy()

clinical_only.to_csv(
    OUTPUT_DIR /
    "clinical_only_breast_records.csv",
    index=False
)


image_only = master[
    master["match_status"] == "Image only"
].copy()

image_only.to_csv(
    OUTPUT_DIR /
    "image_only_breast_records.csv",
    index=False
)


# ============================================================
# 18. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("BREAST-LEVEL MATCHING SUMMARY")
print("=" * 75)

print(
    "\nUnique clinical breast records:",
    len(clinical_counts)
)

print(
    "Unique image breast records:",
    len(image_counts)
)

print(
    "Matched breast records:",
    len(matched)
)

print(
    "Clinical-only breast records:",
    len(clinical_only)
)

print(
    "Image-only breast records:",
    len(image_only)
)


# ============================================================
# 19. MATCH STATUS DISTRIBUTION
# ============================================================

print("\nMatch status:")

print(
    master["match_status"]
    .value_counts()
    .to_string()
)


# ============================================================
# 20. IMAGE COUNT PER MATCHED BREAST
# ============================================================

print("\nImages per matched breast:")

print(
    matched["image_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 21. CLINICAL RECORDS PER MATCHED BREAST
# ============================================================

print("\nClinical records per matched breast:")

print(
    matched[
        "clinical_record_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# 22. CLASSIFICATION DISTRIBUTION
# ============================================================

print("\nClassification distribution among matched breasts:")

classification_distribution = (
    matched[
        "classification"
    ]
    .value_counts(
        dropna=False
    )
)

print(
    classification_distribution
    .to_string()
)

classification_distribution\
    .rename_axis(
        "classification"
    )\
    .reset_index(
        name="breast_count"
    )\
    .to_csv(
        OUTPUT_DIR /
        "matched_breast_classification_distribution.csv",
        index=False
    )


# ============================================================
# 23. L/R DISTRIBUTION
# ============================================================

print("\nMatched breast-side distribution:")

print(
    matched[
        "breast_side"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# 24. PATIENT COUNTS
# ============================================================

print(
    "\nUnique patients represented "
    "in matched breasts:"
)

print(
    matched[
        "patient_id"
    ]
    .nunique()
)


# ============================================================
# 25. SAVE IMAGE FILE LIST PER BREAST
# ============================================================

image_file_lists = (
    images_valid
    .groupby(
        "patient_breast_id"
    )["file_path"]
    .apply(
        lambda x:
        " | ".join(
            sorted(
                x.astype(str)
            )
        )
    )
    .rename(
        "image_files"
    )
)

master_with_files = (
    master
    .set_index(
        "patient_breast_id"
    )
    .join(
        image_file_lists
    )
    .reset_index()
)

master_with_files.to_csv(
    OUTPUT_DIR /
    "breast_level_master_mapping_with_files.csv",
    index=False
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("BREAST-LEVEL MAPPING COMPLETE")
print("=" * 75)

print("\nGenerated:")

print(
    "- breast_level_master_mapping.csv"
)

print(
    "- breast_level_master_mapping_with_files.csv"
)

print(
    "- matched_breast_records.csv"
)

print(
    "- clinical_only_breast_records.csv"
)

print(
    "- image_only_breast_records.csv"
)

print(
    "- matched_breast_classification_distribution.csv"
)

print(
    "\nRaw clinical and DICOM files were NOT modified."
)

print("\nNEXT STEP:")

print(
    "Use the breast-level mapping to define "
    "the final inclusion/exclusion and binary target rules."
)