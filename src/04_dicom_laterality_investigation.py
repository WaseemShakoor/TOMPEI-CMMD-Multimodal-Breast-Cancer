# ============================================================
# TOMPEI-CMMD
# DICOM LATERALITY & VIEW INVESTIGATION
#
# Purpose:
#   Investigate alternative DICOM metadata fields that may
#   identify breast laterality (Left/Right) and mammographic
#   view (CC/MLO).
#
# IMPORTANT:
#   - Reads DICOM metadata only
#   - Does NOT modify raw DICOM files
#   - Does NOT modify clinical data
# ============================================================

from pathlib import Path
import pandas as pd
import pydicom
from collections import Counter

# ============================================================
# 1. PATHS
# ============================================================

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

# ============================================================
# 2. CHECK DIRECTORY
# ============================================================

print("=" * 75)
print("TOMPEI-CMMD DICOM LATERALITY & VIEW INVESTIGATION")
print("=" * 75)

if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_ROOT}"
    )

print("\nScanning DICOM files...")

dicom_files = list(IMAGE_ROOT.rglob("*.dcm"))

print(f"Total DICOM files found: {len(dicom_files):,}")

if not dicom_files:
    raise RuntimeError("No DICOM files found.")

# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def get_value(ds, attribute):
    """
    Safely obtain a DICOM attribute.
    Returns an empty string if unavailable.
    """

    value = getattr(ds, attribute, "")

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# 4. METADATA FIELDS TO INVESTIGATE
# ============================================================

# Standard and potentially useful DICOM attributes.
attributes = [
    "PatientID",
    "StudyDate",
    "SeriesNumber",
    "InstanceNumber",

    # Laterality
    "Laterality",
    "ImageLaterality",

    # Mammography view
    "ViewPosition",

    # Descriptive metadata
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
    "ImageComments",
    "BodyPartExamined",

    # Acquisition metadata
    "AcquisitionNumber",
    "AcquisitionDate",
    "AcquisitionTime",

    # Procedure information
    "PerformedProcedureStepDescription",
    "RequestedProcedureDescription",

    # Identifiers
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
]

# ============================================================
# 5. READ DICOM FILES
# ============================================================

records = []
errors = []

for index, file_path in enumerate(dicom_files, start=1):

    if index % 500 == 0:
        print(
            f"Processed {index:,} / {len(dicom_files):,}"
        )

    try:

        ds = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
            force=True
        )

        record = {
            "file_path": str(file_path),
            "file_name": file_path.name
        }

        for attribute in attributes:
            record[attribute] = get_value(
                ds,
                attribute
            )

        # Dataset inferred from PatientID
        patient_id = record["PatientID"]

        if patient_id.startswith("D1-"):
            dataset = "D1"

        elif patient_id.startswith("D2-"):
            dataset = "D2"

        else:
            dataset = "Unknown"

        record["Dataset"] = dataset

        # Parent folder information can also help us
        # understand the series hierarchy.
        record["parent_folder"] = (
            file_path.parent.name
        )

        record["study_folder"] = (
            file_path.parent.parent.name
            if len(file_path.parents) >= 2
            else ""
        )

        record["patient_folder"] = (
            file_path.parent.parent.parent.name
            if len(file_path.parents) >= 3
            else ""
        )

        records.append(record)

    except Exception as e:

        errors.append({
            "file_path": str(file_path),
            "error": str(e)
        })

# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

print("\nSuccessfully read:", f"{len(df):,}")
print("Read errors:", f"{len(errors):,}")

# ============================================================
# 7. SAVE COMPLETE INVESTIGATION TABLE
# ============================================================

master_output = (
    OUTPUT_DIR /
    "dicom_laterality_metadata_investigation.csv"
)

df.to_csv(
    master_output,
    index=False
)

# ============================================================
# 8. METADATA COMPLETENESS
# ============================================================

metadata_summary = []

for attribute in attributes:

    values = (
        df[attribute]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    available_count = (
        values != ""
    ).sum()

    missing_count = (
        values == ""
    ).sum()

    unique_nonempty = (
        values[
            values != ""
        ].nunique()
    )

    metadata_summary.append({
        "attribute": attribute,
        "available_count": available_count,
        "missing_count": missing_count,
        "available_percentage": round(
            available_count / len(df) * 100,
            2
        ),
        "unique_nonempty_values": unique_nonempty
    })

metadata_summary_df = pd.DataFrame(
    metadata_summary
)

metadata_summary_df.to_csv(
    OUTPUT_DIR /
    "dicom_metadata_completeness.csv",
    index=False
)

print("\n" + "=" * 75)
print("METADATA COMPLETENESS")
print("=" * 75)

print(
    metadata_summary_df.to_string(
        index=False
    )
)

# ============================================================
# 9. INVESTIGATE LATERALITY FIELDS
# ============================================================

print("\n" + "=" * 75)
print("LATERALITY FIELDS")
print("=" * 75)

laterality_fields = [
    "Laterality",
    "ImageLaterality"
]

laterality_records = []

for field in laterality_fields:

    print(f"\n{field}:")

    counts = (
        df[field]
        .replace("", "<MISSING>")
        .value_counts(dropna=False)
    )

    print(counts.to_string())

    for value, count in counts.items():

        laterality_records.append({
            "field": field,
            "value": value,
            "count": count,
            "percentage": round(
                count / len(df) * 100,
                2
            )
        })

pd.DataFrame(
    laterality_records
).to_csv(
    OUTPUT_DIR /
    "dicom_laterality_field_summary.csv",
    index=False
)

# ============================================================
# 10. INVESTIGATE VIEW POSITION
# ============================================================

print("\n" + "=" * 75)
print("VIEW POSITION")
print("=" * 75)

view_counts = (
    df["ViewPosition"]
    .replace("", "<MISSING>")
    .value_counts(dropna=False)
)

print(view_counts.to_string())

view_counts.rename_axis(
    "view_position"
).reset_index(
    name="image_count"
).to_csv(
    OUTPUT_DIR /
    "dicom_view_position_investigation.csv",
    index=False
)

# ============================================================
# 11. DESCRIPTIVE FIELD VALUES
# ============================================================

description_fields = [
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
    "ImageComments",
    "BodyPartExamined",
    "PerformedProcedureStepDescription",
    "RequestedProcedureDescription"
]

description_records = []

print("\n" + "=" * 75)
print("DESCRIPTIVE METADATA")
print("=" * 75)

for field in description_fields:

    nonempty = (
        df[field]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    nonempty = nonempty[
        nonempty != ""
    ]

    print(f"\n{field}")

    if len(nonempty) == 0:

        print("No values available.")

        description_records.append({
            "field": field,
            "value": "<NO VALUES>",
            "count": 0
        })

        continue

    counts = nonempty.value_counts()

    print(counts.head(30).to_string())

    for value, count in counts.items():

        description_records.append({
            "field": field,
            "value": value,
            "count": count
        })

pd.DataFrame(
    description_records
).to_csv(
    OUTPUT_DIR /
    "dicom_descriptive_metadata_summary.csv",
    index=False
)

# ============================================================
# 12. SERIES PER PATIENT
# ============================================================

series_per_patient = (
    df.groupby("PatientID")[
        "SeriesInstanceUID"
    ]
    .nunique()
    .reset_index(
        name="unique_series"
    )
)

series_per_patient.to_csv(
    OUTPUT_DIR /
    "dicom_series_per_patient.csv",
    index=False
)

print("\n" + "=" * 75)
print("SERIES PER PATIENT")
print("=" * 75)

print(
    series_per_patient[
        "unique_series"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# 13. FILE COUNT PER SERIES
# ============================================================

files_per_series = (
    df.groupby(
        [
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID"
        ]
    )
    .size()
    .reset_index(
        name="files_in_series"
    )
)

files_per_series.to_csv(
    OUTPUT_DIR /
    "dicom_files_per_series.csv",
    index=False
)

print("\n" + "=" * 75)
print("FILES PER SERIES")
print("=" * 75)

print(
    files_per_series[
        "files_in_series"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# 14. INSTANCE NUMBER ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("INSTANCE NUMBER")
print("=" * 75)

instance_counts = (
    df["InstanceNumber"]
    .replace("", "<MISSING>")
    .value_counts()
)

print(
    instance_counts.head(30).to_string()
)

instance_counts.rename_axis(
    "instance_number"
).reset_index(
    name="image_count"
).to_csv(
    OUTPUT_DIR /
    "dicom_instance_number_summary.csv",
    index=False
)

# ============================================================
# 15. IMAGE ORDER WITHIN PATIENT
# ============================================================

patient_image_order = df[
    [
        "PatientID",
        "Dataset",
        "file_name",
        "parent_folder",
        "SeriesNumber",
        "InstanceNumber",
        "Laterality",
        "ImageLaterality",
        "ViewPosition",
        "SeriesDescription",
        "ProtocolName",
        "SeriesInstanceUID"
    ]
].copy()

patient_image_order = patient_image_order.sort_values(
    by=[
        "PatientID",
        "SeriesNumber",
        "InstanceNumber",
        "file_name"
    ]
)

patient_image_order.to_csv(
    OUTPUT_DIR /
    "dicom_patient_image_order.csv",
    index=False
)

# ============================================================
# 16. SEARCH TEXT FIELDS FOR L/R/CC/MLO CLUES
# ============================================================

text_fields = [
    "SeriesDescription",
    "StudyDescription",
    "ProtocolName",
    "ImageComments",
    "PerformedProcedureStepDescription",
    "RequestedProcedureDescription"
]

clue_records = []

keywords = [
    "LEFT",
    "RIGHT",
    " L ",
    " R ",
    "CC",
    "MLO",
    "LMLO",
    "RMLO",
    "LCC",
    "RCC"
]

for _, row in df.iterrows():

    combined_text = " | ".join(
        str(row[field])
        for field in text_fields
        if str(row[field]).strip() != ""
    ).upper()

    found_keywords = []

    for keyword in keywords:

        if keyword in combined_text:
            found_keywords.append(
                keyword.strip()
            )

    if found_keywords:

        clue_records.append({
            "PatientID": row["PatientID"],
            "file_name": row["file_name"],
            "found_keywords":
                " + ".join(
                    sorted(
                        set(found_keywords)
                    )
                ),
            "combined_text": combined_text
        })

clue_df = pd.DataFrame(
    clue_records
)

clue_df.to_csv(
    OUTPUT_DIR /
    "dicom_text_laterality_view_clues.csv",
    index=False
)

print("\n" + "=" * 75)
print("TEXT-BASED LATERALITY / VIEW CLUES")
print("=" * 75)

print(
    f"Images containing potential textual clues: "
    f"{len(clue_df):,}"
)

# ============================================================
# 17. SAMPLE PATIENTS WITH 2 AND 4 IMAGES
# ============================================================

patient_counts = (
    df.groupby("PatientID")
    .size()
)

two_image_patients = (
    patient_counts[
        patient_counts == 2
    ]
    .head(10)
    .index
)

four_image_patients = (
    patient_counts[
        patient_counts == 4
    ]
    .head(10)
    .index
)

sample_patients = list(
    two_image_patients
) + list(
    four_image_patients
)

sample_df = patient_image_order[
    patient_image_order[
        "PatientID"
    ].isin(sample_patients)
]

sample_df.to_csv(
    OUTPUT_DIR /
    "dicom_sample_patient_structure.csv",
    index=False
)

# ============================================================
# 18. ERRORS
# ============================================================

if errors:

    pd.DataFrame(
        errors
    ).to_csv(
        OUTPUT_DIR /
        "dicom_laterality_investigation_errors.csv",
        index=False
    )

# ============================================================
# 19. TEXT SUMMARY
# ============================================================

report_file = (
    OUTPUT_DIR /
    "dicom_laterality_investigation_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TOMPEI-CMMD DICOM LATERALITY "
        "& VIEW INVESTIGATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total DICOM files: "
        f"{len(dicom_files):,}\n"
    )

    f.write(
        f"Successfully read: "
        f"{len(df):,}\n"
    )

    f.write(
        f"Read errors: "
        f"{len(errors):,}\n\n"
    )

    f.write(
        "METADATA COMPLETENESS\n"
    )

    f.write(
        metadata_summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nLATERALITY\n"
    )

    for field in laterality_fields:

        f.write(
            f"\n{field}:\n"
        )

        f.write(
            df[field]
            .replace(
                "",
                "<MISSING>"
            )
            .value_counts()
            .to_string()
        )

        f.write("\n")

    f.write(
        "\n\nVIEW POSITION\n"
    )

    f.write(
        view_counts.to_string()
    )

    f.write(
        "\n\nSERIES PER PATIENT\n"
    )

    f.write(
        series_per_patient[
            "unique_series"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    f.write(
        "\n\nFILES PER SERIES\n"
    )

    f.write(
        files_per_series[
            "files_in_series"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    f.write(
        "\n\nNo raw data were modified."
    )

# ============================================================
# 20. FINISH
# ============================================================

print("\n" + "=" * 75)
print("DICOM LATERALITY INVESTIGATION COMPLETE")
print("=" * 75)

print("\nGenerated key files:")

print(
    "- dicom_laterality_metadata_investigation.csv"
)
print(
    "- dicom_metadata_completeness.csv"
)
print(
    "- dicom_laterality_field_summary.csv"
)
print(
    "- dicom_view_position_investigation.csv"
)
print(
    "- dicom_descriptive_metadata_summary.csv"
)
print(
    "- dicom_series_per_patient.csv"
)
print(
    "- dicom_files_per_series.csv"
)
print(
    "- dicom_patient_image_order.csv"
)
print(
    "- dicom_text_laterality_view_clues.csv"
)
print(
    "- dicom_sample_patient_structure.csv"
)
print(
    "- dicom_laterality_investigation_report.txt"
)

print("\nRaw DICOM files were NOT modified.")

print("\nNEXT STEP:")
print(
    "Use the investigation results to determine how "
    "clinical breast-side records map to DICOM images."
)