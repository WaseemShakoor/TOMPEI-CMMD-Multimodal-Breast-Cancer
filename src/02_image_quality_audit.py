# ============================================================
# TOMPEI-CMMD IMAGE / DICOM QUALITY AUDIT
# Master's Dissertation
# ============================================================

from pathlib import Path
import pandas as pd
import pydicom
from collections import Counter
import traceback

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

IMAGE_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\01_Data_Raw\01_Data_Raw\Images_Dataset"
    r"\manifest-1734116293719\CMMD"
)

OUTPUT_DIR = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
    r"\02_Data_Cleaning\audit_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. INITIALISE
# ------------------------------------------------------------

print("=" * 70)
print("TOMPEI-CMMD IMAGE / DICOM QUALITY AUDIT")
print("=" * 70)

print("\nImage root:")
print(IMAGE_ROOT)

if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"\nERROR: Image directory does not exist:\n{IMAGE_ROOT}"
    )

# ------------------------------------------------------------
# 3. FIND ALL DICOM FILES
# ------------------------------------------------------------

print("\nScanning DICOM files...")

dicom_files = list(IMAGE_ROOT.rglob("*.dcm"))

print(f"Total DICOM files found: {len(dicom_files):,}")

if len(dicom_files) == 0:
    raise RuntimeError("No .dcm files were found.")

# ------------------------------------------------------------
# 4. AUDIT VARIABLES
# ------------------------------------------------------------

records = []

read_errors = []

patient_ids = set()
datasets = Counter()

file_extensions = Counter()

# ------------------------------------------------------------
# 5. READ DICOM METADATA
# ------------------------------------------------------------

for index, file_path in enumerate(dicom_files, start=1):

    if index % 500 == 0:
        print(f"Processed {index:,} / {len(dicom_files):,}")

    try:

        ds = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
            force=True
        )

        # ----------------------------------------------------
        # Patient ID
        # ----------------------------------------------------

        patient_id = str(
            getattr(ds, "PatientID", "")
        ).strip()

        if not patient_id:
            patient_id = None

        if patient_id:
            patient_ids.add(patient_id)

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        dataset = None

        if patient_id:
            if patient_id.startswith("D1-"):
                dataset = "D1"
            elif patient_id.startswith("D2-"):
                dataset = "D2"

        if dataset:
            datasets[dataset] += 1

        # ----------------------------------------------------
        # Basic metadata
        # ----------------------------------------------------

        rows = getattr(ds, "Rows", None)
        columns = getattr(ds, "Columns", None)

        modality = str(
            getattr(ds, "Modality", "")
        ).strip()

        body_part = str(
            getattr(ds, "BodyPartExamined", "")
        ).strip()

        laterality = str(
            getattr(ds, "Laterality", "")
        ).strip()

        view_position = str(
            getattr(ds, "ViewPosition", "")
        ).strip()

        photometric = str(
            getattr(ds, "PhotometricInterpretation", "")
        ).strip()

        bits_allocated = getattr(
            ds,
            "BitsAllocated",
            None
        )

        bits_stored = getattr(
            ds,
            "BitsStored",
            None
        )

        samples_per_pixel = getattr(
            ds,
            "SamplesPerPixel",
            None
        )

        study_date = str(
            getattr(ds, "StudyDate", "")
        ).strip()

        study_instance_uid = str(
            getattr(ds, "StudyInstanceUID", "")
        ).strip()

        series_instance_uid = str(
            getattr(ds, "SeriesInstanceUID", "")
        ).strip()

        sop_instance_uid = str(
            getattr(ds, "SOPInstanceUID", "")
        ).strip()

        # ----------------------------------------------------
        # Save record
        # ----------------------------------------------------

        records.append({

            "file_path": str(file_path),

            "file_name": file_path.name,

            "patient_id": patient_id,

            "dataset": dataset,

            "modality": modality,

            "body_part_examined": body_part,

            "laterality": laterality,

            "view_position": view_position,

            "rows": rows,

            "columns": columns,

            "photometric_interpretation": photometric,

            "bits_allocated": bits_allocated,

            "bits_stored": bits_stored,

            "samples_per_pixel": samples_per_pixel,

            "study_date": study_date,

            "study_instance_uid": study_instance_uid,

            "series_instance_uid": series_instance_uid,

            "sop_instance_uid": sop_instance_uid,

        })

    except Exception as e:

        read_errors.append({

            "file_path": str(file_path),

            "error": str(e),

            "traceback": traceback.format_exc()

        })


# ------------------------------------------------------------
# 6. CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(records)

print("\n" + "=" * 70)
print("BASIC IMAGE AUDIT")
print("=" * 70)

print(f"\nSuccessfully read: {len(df):,}")
print(f"Read errors:       {len(read_errors):,}")

print(
    f"Unique patients identified from DICOM metadata: "
    f"{df['patient_id'].nunique():,}"
)

# ------------------------------------------------------------
# 7. DATASET DISTRIBUTION
# ------------------------------------------------------------

print("\nDataset distribution:")

dataset_summary = (
    df["dataset"]
    .value_counts(dropna=False)
    .rename_axis("dataset")
    .reset_index(name="image_count")
)

dataset_summary["percentage"] = (
    dataset_summary["image_count"]
    / len(df)
    * 100
)

print(dataset_summary)

dataset_summary.to_csv(
    OUTPUT_DIR / "image_dataset_distribution.csv",
    index=False
)

# ------------------------------------------------------------
# 8. MODALITY
# ------------------------------------------------------------

print("\nModality distribution:")

modality_summary = (
    df["modality"]
    .value_counts(dropna=False)
    .rename_axis("modality")
    .reset_index(name="image_count")
)

print(modality_summary)

modality_summary.to_csv(
    OUTPUT_DIR / "image_modality_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 9. LATERALITY
# ------------------------------------------------------------

print("\nLaterality distribution:")

laterality_summary = (
    df["laterality"]
    .value_counts(dropna=False)
    .rename_axis("laterality")
    .reset_index(name="image_count")
)

laterality_summary["percentage"] = (
    laterality_summary["image_count"]
    / len(df)
    * 100
)

print(laterality_summary)

laterality_summary.to_csv(
    OUTPUT_DIR / "image_laterality_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 10. VIEW POSITION
# ------------------------------------------------------------

print("\nView position distribution:")

view_summary = (
    df["view_position"]
    .value_counts(dropna=False)
    .rename_axis("view_position")
    .reset_index(name="image_count")
)

print(view_summary)

view_summary.to_csv(
    OUTPUT_DIR / "image_view_position_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 11. IMAGE DIMENSIONS
# ------------------------------------------------------------

print("\nImage dimensions:")

dimension_summary = (
    df.groupby(
        ["rows", "columns"],
        dropna=False
    )
    .size()
    .reset_index(name="image_count")
    .sort_values(
        "image_count",
        ascending=False
    )
)

print(dimension_summary.head(20))

dimension_summary.to_csv(
    OUTPUT_DIR / "image_dimension_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 12. PHOTOMETRIC INTERPRETATION
# ------------------------------------------------------------

print("\nPhotometric interpretation:")

photometric_summary = (
    df["photometric_interpretation"]
    .value_counts(dropna=False)
    .rename_axis("photometric_interpretation")
    .reset_index(name="image_count")
)

print(photometric_summary)

photometric_summary.to_csv(
    OUTPUT_DIR / "image_photometric_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 13. IMAGE COUNT PER PATIENT
# ------------------------------------------------------------

print("\nImages per patient:")

patient_image_counts = (
    df.groupby(
        "patient_id",
        dropna=False
    )
    .size()
    .reset_index(name="image_count")
)

print(
    patient_image_counts["image_count"]
    .describe()
)

patient_image_counts.to_csv(
    OUTPUT_DIR / "image_count_per_patient.csv",
    index=False
)

# ------------------------------------------------------------
# 14. PATIENT / DATASET SUMMARY
# ------------------------------------------------------------

patient_dataset_summary = (
    df.groupby(
        ["dataset", "patient_id"],
        dropna=False
    )
    .size()
    .reset_index(name="image_count")
)

patient_dataset_summary.to_csv(
    OUTPUT_DIR / "patient_dataset_image_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 15. DUPLICATE SOP INSTANCE UID
# ------------------------------------------------------------

print("\nChecking duplicate SOP Instance UIDs...")

uid_counts = (
    df["sop_instance_uid"]
    .value_counts()
)

duplicate_uids = uid_counts[
    uid_counts > 1
]

print(
    f"Duplicate SOP Instance UIDs: "
    f"{len(duplicate_uids):,}"
)

duplicate_uid_df = (
    duplicate_uids
    .rename_axis("sop_instance_uid")
    .reset_index(name="count")
)

duplicate_uid_df.to_csv(
    OUTPUT_DIR / "duplicate_sop_instance_uids.csv",
    index=False
)

# ------------------------------------------------------------
# 16. MISSING METADATA
# ------------------------------------------------------------

metadata_columns = [
    "patient_id",
    "dataset",
    "modality",
    "laterality",
    "view_position",
    "rows",
    "columns",
    "photometric_interpretation",
    "bits_allocated",
    "bits_stored",
    "samples_per_pixel",
    "study_instance_uid",
    "series_instance_uid",
    "sop_instance_uid"
]

missing_metadata = []

for column in metadata_columns:

    missing = df[column].isna().sum()

    if df[column].dtype == "object":
        missing += (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    missing_metadata.append({

        "column": column,

        "missing_count": int(missing),

        "missing_percentage":
            round(
                missing / len(df) * 100,
                2
            )
    })

missing_metadata_df = pd.DataFrame(
    missing_metadata
)

print("\nMissing DICOM metadata:")

print(
    missing_metadata_df
    .sort_values(
        "missing_percentage",
        ascending=False
    )
)

missing_metadata_df.to_csv(
    OUTPUT_DIR / "image_missing_metadata.csv",
    index=False
)

# ------------------------------------------------------------
# 17. READ ERRORS
# ------------------------------------------------------------

if read_errors:

    error_df = pd.DataFrame(read_errors)

    error_df.to_csv(
        OUTPUT_DIR / "image_read_errors.csv",
        index=False
    )

# ------------------------------------------------------------
# 18. MASTER IMAGE INVENTORY
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "image_master_inventory.csv",
    index=False
)

# ------------------------------------------------------------
# 19. TEXT SUMMARY
# ------------------------------------------------------------

summary_file = (
    OUTPUT_DIR /
    "image_quality_audit_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TOMPEI-CMMD IMAGE / DICOM QUALITY AUDIT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total DICOM files found: "
        f"{len(dicom_files):,}\n"
    )

    f.write(
        f"Successfully read: "
        f"{len(df):,}\n"
    )

    f.write(
        f"Read errors: "
        f"{len(read_errors):,}\n"
    )

    f.write(
        f"Unique patients: "
        f"{df['patient_id'].nunique():,}\n"
    )

    f.write("\nDataset distribution:\n")
    f.write(
        dataset_summary.to_string(
            index=False
        )
    )

    f.write("\n\nModality distribution:\n")
    f.write(
        modality_summary.to_string(
            index=False
        )
    )

    f.write("\n\nLaterality distribution:\n")
    f.write(
        laterality_summary.to_string(
            index=False
        )
    )

    f.write("\n\nView position distribution:\n")
    f.write(
        view_summary.to_string(
            index=False
        )
    )

# ------------------------------------------------------------
# 20. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMAGE AUDIT COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print("-", file.name)

print("\nNo changes were made to the raw DICOM files.")

print("\nNext step:")
print("Clinical ↔ Image patient-level matching audit.")