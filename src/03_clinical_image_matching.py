# ============================================================
# TOMPEI-CMMD CLINICAL <-> IMAGE PATIENT MATCHING AUDIT
# Master's Dissertation
#
# Purpose:
#   1. Compare clinical patient IDs with image patient IDs
#   2. Identify matched, clinical-only and image-only patients
#   3. Preserve all clinical classifications
#   4. Identify patients with multiple/conflicting classifications
#   5. Summarise image counts per patient
#   6. Create a master patient-level matching table
#
# IMPORTANT:
#   - Raw clinical Excel file is NOT modified
#   - Raw DICOM files are NOT modified
#   - No patients are deleted
#   - No diagnosis is automatically assigned when classifications conflict
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

CLINICAL_FILE = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\01_Data_Raw\01_Data_Raw\Tabular_Dataset"
    r"\TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
)

IMAGE_INVENTORY = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\02_Data_Cleaning\audit_outputs"
    r"\image_master_inventory.csv"
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
# 2. START
# ============================================================

print("=" * 70)
print("TOMPEI-CMMD CLINICAL <-> IMAGE PATIENT MATCHING AUDIT")
print("=" * 70)


# ============================================================
# 3. CHECK FILES
# ============================================================

if not CLINICAL_FILE.exists():

    raise FileNotFoundError(
        f"\nClinical dataset not found:\n{CLINICAL_FILE}"
    )

if not IMAGE_INVENTORY.exists():

    raise FileNotFoundError(
        f"\nImage inventory not found:\n{IMAGE_INVENTORY}"
    )


# ============================================================
# 4. LOAD CLINICAL DATA
# ============================================================

print("\nLoading clinical dataset...")

clinical = pd.read_excel(
    CLINICAL_FILE,
    sheet_name="Imaging Diagnosis Details Sheet"
)

print(
    f"Clinical rows loaded: {len(clinical):,}"
)

print(
    f"Clinical columns loaded: {len(clinical.columns):,}"
)


# ============================================================
# 5. CHECK REQUIRED CLINICAL COLUMNS
# ============================================================

required_clinical_columns = [
    "ID",
    "LeftRight",
    "classification"
]

missing_columns = [
    column
    for column in required_clinical_columns
    if column not in clinical.columns
]

if missing_columns:

    raise ValueError(
        "\nThe following required clinical columns are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 6. LOAD IMAGE INVENTORY
# ============================================================

print("\nLoading image inventory...")

images = pd.read_csv(
    IMAGE_INVENTORY
)

print(
    f"Image records loaded: {len(images):,}"
)


# ============================================================
# 7. CHECK IMAGE PATIENT ID
# ============================================================

if "patient_id" not in images.columns:

    raise ValueError(
        "\n'image_master_inventory.csv' does not contain "
        "'patient_id'."
    )


# ============================================================
# 8. NORMALISE PATIENT IDS
# ============================================================

print("\nNormalising patient IDs...")

clinical["patient_id"] = (
    clinical["ID"]
    .astype("string")
    .str.strip()
)

images["patient_id"] = (
    images["patient_id"]
    .astype("string")
    .str.strip()
)


# ============================================================
# 9. REMOVE EMPTY IDS ONLY FOR MATCHING
# ============================================================

clinical_valid = clinical[
    clinical["patient_id"].notna()
    & clinical["patient_id"].ne("")
].copy()

images_valid = images[
    images["patient_id"].notna()
    & images["patient_id"].ne("")
].copy()


# ============================================================
# 10. UNIQUE PATIENT IDS
# ============================================================

clinical_patient_ids = set(
    clinical_valid["patient_id"]
    .unique()
)

image_patient_ids = set(
    images_valid["patient_id"]
    .unique()
)


# ============================================================
# 11. SET COMPARISON
# ============================================================

matched_patients = (
    clinical_patient_ids
    & image_patient_ids
)

clinical_only_patients = (
    clinical_patient_ids
    - image_patient_ids
)

image_only_patients = (
    image_patient_ids
    - clinical_patient_ids
)


# ============================================================
# 12. BASIC MATCHING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PATIENT MATCHING SUMMARY")
print("=" * 70)

print(
    f"\nUnique clinical patients: "
    f"{len(clinical_patient_ids):,}"
)

print(
    f"Unique image patients: "
    f"{len(image_patient_ids):,}"
)

print(
    f"Matched patients: "
    f"{len(matched_patients):,}"
)

print(
    f"Clinical-only patients: "
    f"{len(clinical_only_patients):,}"
)

print(
    f"Image-only patients: "
    f"{len(image_only_patients):,}"
)


# ============================================================
# 13. MATCHING PERCENTAGES
# ============================================================

clinical_match_percentage = (
    len(matched_patients)
    / len(clinical_patient_ids)
    * 100
)

image_match_percentage = (
    len(matched_patients)
    / len(image_patient_ids)
    * 100
)

print(
    f"\nPercentage of clinical patients with images: "
    f"{clinical_match_percentage:.2f}%"
)

print(
    f"Percentage of image patients with clinical data: "
    f"{image_match_percentage:.2f}%"
)


# ============================================================
# 14. CLINICAL RECORD COUNT PER PATIENT
# ============================================================

clinical_record_counts = (
    clinical_valid
    .groupby("patient_id")
    .size()
    .rename("clinical_record_count")
)


# ============================================================
# 15. IMAGE COUNT PER PATIENT
# ============================================================

image_counts = (
    images_valid
    .groupby("patient_id")
    .size()
    .rename("image_count")
)


# ============================================================
# 16. CLINICAL CLASSIFICATIONS PER PATIENT
# ============================================================

def unique_classifications(series):

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return sorted(
        values.unique()
    )


classification_by_patient = (
    clinical_valid
    .groupby("patient_id")["classification"]
    .apply(unique_classifications)
)


# ============================================================
# 17. CONVERT CLASSIFICATION LIST TO TEXT
# ============================================================

classification_text = (
    classification_by_patient
    .apply(
        lambda x: " + ".join(x)
        if len(x) > 0
        else ""
    )
    .rename("clinical_classifications")
)


# ============================================================
# 18. NUMBER OF CLASSIFICATIONS
# ============================================================

classification_count = (
    classification_by_patient
    .apply(len)
    .rename("number_of_classifications")
)


# ============================================================
# 19. CONFLICT FLAG
# ============================================================

classification_conflict = (
    classification_count
    > 1
).rename(
    "classification_conflict"
)


# ============================================================
# 20. DATASET INFORMATION
# ============================================================

dataset_by_patient = (
    clinical_valid
    .assign(
        dataset=clinical_valid["patient_id"]
        .astype(str)
        .str[:2]
    )
    .groupby("patient_id")["dataset"]
    .apply(
        lambda x: " + ".join(
            sorted(x.dropna().unique())
        )
    )
    .rename("clinical_dataset")
)


# ============================================================
# 21. IMAGE DATASET INFORMATION
# ============================================================

if "dataset" in images_valid.columns:

    image_dataset_by_patient = (
        images_valid
        .groupby("patient_id")["dataset"]
        .apply(
            lambda x: " + ".join(
                sorted(
                    x.dropna()
                    .astype(str)
                    .unique()
                )
            )
        )
        .rename("image_dataset")

    )

else:

    image_dataset_by_patient = pd.Series(
        dtype="object",
        name="image_dataset"
    )


# ============================================================
# 22. LATERALITY / IMAGE INFORMATION
# ============================================================

if "laterality" in images_valid.columns:

    image_laterality = (
        images_valid
        .groupby("patient_id")["laterality"]
        .apply(
            lambda x: " + ".join(
                sorted(
                    x.dropna()
                    .astype(str)
                    .unique()
                )
            )
        )
        .rename("image_laterality")
    )

else:

    image_laterality = pd.Series(
        dtype="object",
        name="image_laterality"
    )


# ============================================================
# 23. VIEW POSITION
# ============================================================

if "view_position" in images_valid.columns:

    image_views = (
        images_valid
        .groupby("patient_id")["view_position"]
        .apply(
            lambda x: " + ".join(
                sorted(
                    x.dropna()
                    .astype(str)
                    .unique()
                )
            )
        )
        .rename("image_view_positions")
    )

else:

    image_views = pd.Series(
        dtype="object",
        name="image_view_positions"
    )


# ============================================================
# 24. CREATE MASTER PATIENT TABLE
# ============================================================

all_patient_ids = sorted(
    clinical_patient_ids
    | image_patient_ids
)

master = pd.DataFrame({
    "patient_id": all_patient_ids
})

master = master.set_index(
    "patient_id"
)


# ------------------------------------------------------------
# Clinical information
# ------------------------------------------------------------

master = master.join(
    clinical_record_counts
)

master = master.join(
    classification_text
)

master = master.join(
    classification_count
)

master = master.join(
    classification_conflict
)

master = master.join(
    dataset_by_patient
)


# ------------------------------------------------------------
# Image information
# ------------------------------------------------------------

master = master.join(
    image_counts
)

master = master.join(
    image_dataset_by_patient
)

master = master.join(
    image_laterality
)

master = master.join(
    image_views
)


# ============================================================
# 25. FILL COUNTS
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

master["number_of_classifications"] = (
    master["number_of_classifications"]
    .fillna(0)
    .astype(int)
)

master["classification_conflict"] = (
    master["classification_conflict"]
    .fillna(False)
)


# ============================================================
# 26. MATCH STATUS
# ============================================================

def determine_match_status(row):

    has_clinical = (
        row["clinical_record_count"] > 0
    )

    has_image = (
        row["image_count"] > 0
    )

    if has_clinical and has_image:
        return "Matched"

    elif has_clinical:
        return "Clinical only"

    elif has_image:
        return "Image only"

    return "Unknown"


master["match_status"] = master.apply(
    determine_match_status,
    axis=1
)


# ============================================================
# 27. PATIENT-LEVEL CLASSIFICATION TYPE
# ============================================================

def classification_type(row):

    value = row["clinical_classifications"]

    if pd.isna(value) or value == "":
        return "No clinical classification"

    if row["classification_conflict"]:
        return "Multiple classifications"

    return "Single classification"


master["classification_type"] = master.apply(
    classification_type,
    axis=1
)


# ============================================================
# 28. REORDER COLUMNS
# ============================================================

master = master.reset_index()

preferred_columns = [
    "patient_id",
    "match_status",
    "clinical_record_count",
    "image_count",
    "clinical_dataset",
    "image_dataset",
    "clinical_classifications",
    "number_of_classifications",
    "classification_type",
    "classification_conflict",
    "image_laterality",
    "image_view_positions"
]

remaining_columns = [
    column
    for column in master.columns
    if column not in preferred_columns
]

master = master[
    preferred_columns
    + remaining_columns
]


# ============================================================
# 29. SAVE MASTER MATCHING TABLE
# ============================================================

master_file = (
    OUTPUT_DIR /
    "clinical_image_patient_master.csv"
)

master.to_csv(
    master_file,
    index=False
)

print(
    f"\nMaster patient table saved:\n{master_file}"
)


# ============================================================
# 30. SAVE MATCHED PATIENTS
# ============================================================

matched_df = master[
    master["match_status"] == "Matched"
].copy()

matched_file = (
    OUTPUT_DIR /
    "matched_patients.csv"
)

matched_df.to_csv(
    matched_file,
    index=False
)


# ============================================================
# 31. SAVE CLINICAL-ONLY PATIENTS
# ============================================================

clinical_only_df = master[
    master["match_status"] == "Clinical only"
].copy()

clinical_only_file = (
    OUTPUT_DIR /
    "clinical_only_patients.csv"
)

clinical_only_df.to_csv(
    clinical_only_file,
    index=False
)


# ============================================================
# 32. SAVE IMAGE-ONLY PATIENTS
# ============================================================

image_only_df = master[
    master["match_status"] == "Image only"
].copy()

image_only_file = (
    OUTPUT_DIR /
    "image_only_patients.csv"
)

image_only_df.to_csv(
    image_only_file,
    index=False
)


# ============================================================
# 33. SAVE CONFLICTING PATIENTS
# ============================================================

conflict_df = master[
    master["classification_conflict"] == True
].copy()

conflict_file = (
    OUTPUT_DIR /
    "patients_with_multiple_classifications.csv"
)

conflict_df.to_csv(
    conflict_file,
    index=False
)


# ============================================================
# 34. CLASSIFICATION DISTRIBUTION AMONG MATCHED PATIENTS
# ============================================================

matched_classification_distribution = (
    matched_df[
        "clinical_classifications"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "clinical_classifications"
    )
    .reset_index(
        name="patient_count"
    )
)

matched_classification_distribution.to_csv(
    OUTPUT_DIR /
    "matched_patient_classification_distribution.csv",
    index=False
)


# ============================================================
# 35. MATCHING SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "measure": [

        "Unique clinical patients",

        "Unique image patients",

        "Matched patients",

        "Clinical-only patients",

        "Image-only patients",

        "Clinical patients with images (%)",

        "Image patients with clinical data (%)",

        "Matched patients with multiple classifications",

        "Matched patients with single classification",

    ],

    "value": [

        len(clinical_patient_ids),

        len(image_patient_ids),

        len(matched_patients),

        len(clinical_only_patients),

        len(image_only_patients),

        round(
            clinical_match_percentage,
            2
        ),

        round(
            image_match_percentage,
            2
        ),

        int(
            conflict_df[
                "match_status"
            ]
            .eq("Matched")
            .sum()
        ),

        int(
            matched_df[
                "classification_conflict"
            ]
            .eq(False)
            .sum()
        )

    ]

})


# ============================================================
# 36. SAVE SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR /
    "clinical_image_matching_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 37. PRINT MATCHING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(
    f"\nUnique clinical patients: "
    f"{len(clinical_patient_ids):,}"
)

print(
    f"Unique image patients: "
    f"{len(image_patient_ids):,}"
)

print(
    f"Matched patients: "
    f"{len(matched_patients):,}"
)

print(
    f"Clinical-only patients: "
    f"{len(clinical_only_patients):,}"
)

print(
    f"Image-only patients: "
    f"{len(image_only_patients):,}"
)

print(
    f"\nClinical patients with images: "
    f"{clinical_match_percentage:.2f}%"
)

print(
    f"Image patients with clinical data: "
    f"{image_match_percentage:.2f}%"
)


# ============================================================
# 38. MATCH STATUS DISTRIBUTION
# ============================================================

print("\nMatch status distribution:")

print(
    master["match_status"]
    .value_counts()
)


# ============================================================
# 39. CLASSIFICATION DISTRIBUTION
# ============================================================

print(
    "\nMatched patient classification combinations:"
)

print(
    matched_classification_distribution
    .to_string(index=False)
)


# ============================================================
# 40. CONFLICTING PATIENTS
# ============================================================

matched_conflicts = matched_df[
    matched_df["classification_conflict"]
]

print(
    "\nMatched patients with multiple classifications: "
    f"{len(matched_conflicts):,}"
)

if len(matched_conflicts) > 0:

    print(
        "\nClassification combinations among "
        "matched conflicting patients:"
    )

    print(
        matched_conflicts[
            "clinical_classifications"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# 41. SAVE TEXT REPORT
# ============================================================

report_file = (
    OUTPUT_DIR /
    "clinical_image_matching_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TOMPEI-CMMD CLINICAL <-> IMAGE "
        "PATIENT MATCHING AUDIT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Unique clinical patients: "
        f"{len(clinical_patient_ids):,}\n"
    )

    f.write(
        f"Unique image patients: "
        f"{len(image_patient_ids):,}\n"
    )

    f.write(
        f"Matched patients: "
        f"{len(matched_patients):,}\n"
    )

    f.write(
        f"Clinical-only patients: "
        f"{len(clinical_only_patients):,}\n"
    )

    f.write(
        f"Image-only patients: "
        f"{len(image_only_patients):,}\n"
    )

    f.write(
        f"\nClinical patients with images: "
        f"{clinical_match_percentage:.2f}%\n"
    )

    f.write(
        f"Image patients with clinical data: "
        f"{image_match_percentage:.2f}%\n"
    )

    f.write(
        f"\nMatched patients with multiple "
        f"classifications: "
        f"{len(matched_conflicts):,}\n"
    )

    f.write(
        "\n\nMatched classification combinations:\n"
    )

    f.write(
        matched_classification_distribution
        .to_string(index=False)
    )

    f.write(
        "\n\nNo changes were made to the raw "
        "clinical or DICOM datasets.\n"
    )


# ============================================================
# 42. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CLINICAL <-> IMAGE MATCHING AUDIT COMPLETE")
print("=" * 70)

print("\nGenerated files:")

output_files = [
    master_file,
    matched_file,
    clinical_only_file,
    image_only_file,
    conflict_file,
    OUTPUT_DIR /
    "matched_patient_classification_distribution.csv",
    summary_file,
    report_file
]

for file in output_files:

    print(
        f"- {file.name}"
    )

print(
    "\nRaw clinical and image data were not modified."
)

print(
    "\nNEXT STEP:"
)

print(
    "Review the matching results before performing "
    "clinical data cleaning or model preparation."
)