# ============================================================
# TOMPEI-CMMD
# Clinical Data Quality Audit
#
# Purpose:
#   Examine the raw clinical dataset before any cleaning.
#
# IMPORTANT:
#   This script DOES NOT modify the raw Excel dataset.
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

from pathlib import Path

# Repository root:
# TOMPEI-CMMD-Multimodal-Breast-Cancer/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw data are intentionally NOT included in the GitHub repository.
# Download TOMPEI-CMMD separately and place the clinical workbook
# in the location below, or change this path for your environment.
CLINICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
)

# Audit outputs
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "data_quality"
    / "clinical"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not CLINICAL_FILE.exists():
    raise FileNotFoundError(
        "\nClinical dataset not found.\n"
        f"Expected location:\n{CLINICAL_FILE}\n\n"
        "The TOMPEI-CMMD dataset is not distributed with this repository. "
        "Download the dataset separately and place the clinical Excel file "
        "in data/raw/ before running this script."
    )

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("TOMPEI-CMMD CLINICAL DATA QUALITY AUDIT")
print("=" * 70)

print("\nLoading dataset...")

try:

    excel_file = pd.ExcelFile(CLINICAL_FILE)

    print("\nAvailable sheets:")
    for sheet in excel_file.sheet_names:
        print(" -", sheet)

except Exception as e:

    print("\nERROR loading Excel file:")
    print(e)
    raise


# ============================================================
# 3. SELECT CLINICAL SHEET
# ============================================================

SHEET_NAME = "Imaging Diagnosis Details Sheet"

if SHEET_NAME not in excel_file.sheet_names:

    print("\nWARNING:")
    print(f"Sheet '{SHEET_NAME}' was not found.")

    print("\nPlease check the sheet names above.")

    raise ValueError(
        f"Sheet '{SHEET_NAME}' not found."
    )


df = pd.read_excel(
    CLINICAL_FILE,
    sheet_name=SHEET_NAME
)

print("\nDataset loaded successfully.")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 4. DISPLAY COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):

    print(f"{i:02d}. {column}")


# ============================================================
# 5. BASIC DATASET OVERVIEW
# ============================================================

overview = pd.DataFrame({
    "Measure": [
        "Total rows",
        "Total columns",
        "Unique patient IDs",
        "Duplicate rows",
        "Completely empty rows",
        "Completely empty columns"
    ],
    "Value": [
        len(df),
        len(df.columns),
        df["ID"].nunique()
        if "ID" in df.columns else np.nan,
        df.duplicated().sum(),
        df.isna().all(axis=1).sum(),
        df.isna().all(axis=0).sum()
    ]
})

overview.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clinical_dataset_overview.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(overview.to_string(index=False))


# ============================================================
# 6. COLUMN-LEVEL QUALITY
# ============================================================

column_quality = pd.DataFrame({
    "column": df.columns,
    "data_type": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "non_null_count": [
        df[col].notna().sum()
        for col in df.columns
    ],
    "missing_count": [
        df[col].isna().sum()
        for col in df.columns
    ],
    "missing_percentage": [
        round(df[col].isna().mean() * 100, 2)
        for col in df.columns
    ],
    "unique_values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

column_quality.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clinical_column_quality.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("COLUMN QUALITY")
print("=" * 70)

print(
    column_quality.to_string(index=False)
)


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": [
        df[col].isna().sum()
        for col in df.columns
    ],
    "missing_percentage": [
        round(df[col].isna().mean() * 100, 2)
        for col in df.columns
    ]
})

missing = missing.sort_values(
    by="missing_percentage",
    ascending=False
)

missing.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clinical_missing_values.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

print(
    missing.to_string(index=False)
)


# ============================================================
# 8. EXACT DUPLICATE ROWS
# ============================================================

duplicate_rows = df[
    df.duplicated(
        keep=False
    )
].copy()

duplicate_rows.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clinical_duplicate_rows.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("DUPLICATE ROW ANALYSIS")
print("=" * 70)

print(
    "Number of exact duplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 9. PATIENT-LEVEL RECORD COUNTS
# ============================================================

if "ID" in df.columns:

    patient_counts = (
        df.groupby("ID")
        .size()
        .reset_index(
            name="record_count"
        )
    )

    patient_counts["record_type"] = np.where(
        patient_counts["record_count"] == 1,
        "One record",
        "Multiple records"
    )

    patient_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_patient_record_counts.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL RECORD STRUCTURE")
    print("=" * 70)

    print(
        patient_counts[
            "record_count"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nUnique patients:",
        patient_counts["ID"].nunique()
    )

    print(
        "Patients with multiple records:",
        (
            patient_counts["record_count"] > 1
        ).sum()
    )


# ============================================================
# 10. CLASSIFICATION DISTRIBUTION
# ============================================================

if "classification" in df.columns:

    classification = (
        df["classification"]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    classification.columns = [
        "classification",
        "record_count"
    ]

    classification["percentage"] = (
        classification["record_count"]
        / len(df)
        * 100
    ).round(2)

    classification.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_classification_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION DISTRIBUTION")
    print("=" * 70)

    print(
        classification.to_string(
            index=False
        )
    )


# ============================================================
# 11. DATASET (D1 / D2) DISTRIBUTION
# ============================================================

if "ID" in df.columns:

    df["Dataset"] = (
        df["ID"]
        .astype(str)
        .str[:2]
    )

    dataset_distribution = (
        df["Dataset"]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    dataset_distribution.columns = [
        "dataset",
        "record_count"
    ]

    dataset_distribution["percentage"] = (
        dataset_distribution["record_count"]
        / len(df)
        * 100
    ).round(2)

    dataset_distribution.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_D1_D2_distribution.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("D1 / D2 DISTRIBUTION")
    print("=" * 70)

    print(
        dataset_distribution.to_string(
            index=False
        )
    )


# ============================================================
# 12. D1 / D2 × CLASSIFICATION
# ============================================================

if (
    "Dataset" in df.columns
    and "classification" in df.columns
):

    dataset_classification = pd.crosstab(
        df["Dataset"],
        df["classification"],
        margins=True
    )

    dataset_classification.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_D1_D2_classification.csv"
        )
    )

    print("\n" + "=" * 70)
    print("D1 / D2 × CLASSIFICATION")
    print("=" * 70)

    print(
        dataset_classification.to_string()
    )


# ============================================================
# 13. CATEGORICAL VARIABLE ANALYSIS
# ============================================================

categorical_records = []

for column in df.columns:

    if df[column].dtype == "object":

        value_counts = (
            df[column]
            .value_counts(
                dropna=False
            )
        )

        for value, count in value_counts.items():

            if pd.isna(value):
                value_display = "<MISSING>"
            else:
                value_display = str(value)

            categorical_records.append({

                "column": column,

                "value": value_display,

                "count": int(count),

                "percentage": round(
                    count / len(df) * 100,
                    2
                )
            })


categorical_summary = pd.DataFrame(
    categorical_records
)

categorical_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "clinical_categorical_summary.csv"
    ),
    index=False
)


# ============================================================
# 14. NUMERICAL VARIABLE ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

if len(numeric_columns) > 0:

    numeric_summary = (
        df[numeric_columns]
        .describe()
        .T
    )

    numeric_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_numeric_summary.csv"
        )
    )

    print("\n" + "=" * 70)
    print("NUMERICAL VARIABLE SUMMARY")
    print("=" * 70)

    print(
        numeric_summary.to_string()
    )

else:

    print(
        "\nNo numerical columns detected."
    )


# ============================================================
# 15. BREAST-SIDE ANALYSIS
# ============================================================

if "LeftRight" in df.columns:

    side_summary = (
        df["LeftRight"]
        .value_counts(
            dropna=False
        )
        .reset_index()
    )

    side_summary.columns = [
        "breast_side",
        "record_count"
    ]

    side_summary["percentage"] = (
        side_summary["record_count"]
        / len(df)
        * 100
    ).round(2)

    side_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_breast_side_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("BREAST-SIDE DISTRIBUTION")
    print("=" * 70)

    print(
        side_summary.to_string(
            index=False
        )
    )


# ============================================================
# 16. PATIENT-LEVEL CLASSIFICATION COMBINATIONS
# ============================================================

if (
    "ID" in df.columns
    and "classification" in df.columns
):

    patient_classification = (
        df.groupby("ID")["classification"]
        .apply(
            lambda x:
            " + ".join(
                sorted(
                    set(
                        x.dropna()
                    )
                )
            )
        )
        .reset_index()
    )

    patient_classification.columns = [
        "patient_id",
        "classification_combination"
    ]

    combination_counts = (
        patient_classification[
            "classification_combination"
        ]
        .value_counts()
        .reset_index()
    )

    combination_counts.columns = [
        "classification_combination",
        "patient_count"
    ]

    combination_counts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_patient_classification_combinations.csv"
        ),
        index=False
    )

    patient_classification.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_patient_level_audit.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print(
        "PATIENT-LEVEL CLASSIFICATION COMBINATIONS"
    )
    print("=" * 70)

    print(
        combination_counts.to_string(
            index=False
        )
    )


# ============================================================
# 17. MISSING PATIENT IDs
# ============================================================

if "ID" in df.columns:

    missing_ids = df[
        df["ID"].isna()
    ].copy()

    missing_ids.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clinical_missing_patient_ids.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("PATIENT ID CHECK")
    print("=" * 70)

    print(
        "Rows with missing patient ID:",
        len(missing_ids)
    )


# ============================================================
# 18. SAVE A TEXT SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "clinical_quality_audit_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TOMPEI-CMMD CLINICAL DATA QUALITY AUDIT\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total rows: {len(df)}\n"
    )

    f.write(
        f"Total columns: {len(df.columns)}\n"
    )

    if "ID" in df.columns:

        f.write(
            f"Unique patients: "
            f"{df['ID'].nunique()}\n"
        )

    f.write(
        f"Exact duplicate rows: "
        f"{df.duplicated().sum()}\n"
    )

    f.write(
        f"Completely empty rows: "
        f"{df.isna().all(axis=1).sum()}\n"
    )

    f.write(
        f"Completely empty columns: "
        f"{df.isna().all(axis=0).sum()}\n"
    )


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nNo changes were made to the raw Excel dataset.")

print("\nAudit files saved to:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    if (
        filename.endswith(".csv")
        or filename.endswith(".txt")
    ):

        print(" -", filename)

print("\nNext step:")
print(
    "Review the audit results before making any cleaning decisions."
)

print("=" * 70)