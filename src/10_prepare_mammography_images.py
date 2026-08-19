# ============================================================
# TOMPEI-CMMD
# MAMMOGRAPHY IMAGE PREPARATION
#
# Purpose:
#   - Read original DICOM mammograms
#   - Preserve raw DICOM files
#   - Apply safe intensity normalisation
#   - Preserve aspect ratio
#   - Pad to 512 x 512
#   - Save processed grayscale PNG images
#   - Preserve Train / Validation / Test assignments
#   - Create a final image manifest for DL and multimodal models
#
# IMPORTANT:
#   Raw DICOM files are NEVER modified.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import pydicom

from PIL import Image

# Optional DICOM LUT utilities
try:
    from pydicom.pixels import apply_modality_lut
except ImportError:
    apply_modality_lut = None


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

# This file already contains:
# patient_id
# patient_breast_id
# target
# image_file_1
# image_file_2
# split
INPUT_SPLIT_FILE = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "splits"
    / "all_breast_level_with_split.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "mammography_512"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "05_Results"
    / "image_preprocessing"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

OUTPUT_SIZE = 512

# Robust percentile clipping.
# This reduces the influence of extreme pixel outliers.
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

# Do not overwrite already processed PNG files.
# Useful if the script is interrupted and re-run.
OVERWRITE = False


# ============================================================
# 3. LOAD FIXED DATA SPLITS
# ============================================================

print("=" * 78)
print("TOMPEI-CMMD MAMMOGRAPHY IMAGE PREPARATION")
print("=" * 78)

if not INPUT_SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{INPUT_SPLIT_FILE}"
    )

df = pd.read_csv(
    INPUT_SPLIT_FILE
)

print(
    f"\nBreast records loaded: "
    f"{len(df):,}"
)

print(
    f"Unique patients: "
    f"{df['patient_id'].nunique():,}"
)


# ============================================================
# 4. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "patient_breast_id",
    "patient_id",
    "breast_side",
    "target",
    "split",
    "image_file_1",
    "image_file_2"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 5. CHECK SPLIT VALUES
# ============================================================

valid_splits = {
    "train",
    "validation",
    "test"
}

actual_splits = set(
    df["split"]
    .dropna()
    .astype(str)
    .str.strip()
)

unexpected_splits = (
    actual_splits
    - valid_splits
)

if unexpected_splits:
    raise ValueError(
        f"Unexpected split labels: "
        f"{unexpected_splits}"
    )


# ============================================================
# 6. CHECK PATIENT OVERLAP AGAIN
# ============================================================

train_patients = set(
    df.loc[
        df["split"] == "train",
        "patient_id"
    ]
)

validation_patients = set(
    df.loc[
        df["split"] == "validation",
        "patient_id"
    ]
)

test_patients = set(
    df.loc[
        df["split"] == "test",
        "patient_id"
    ]
)

assert len(
    train_patients
    & validation_patients
) == 0

assert len(
    train_patients
    & test_patients
) == 0

assert len(
    validation_patients
    & test_patients
) == 0

print(
    "\nPatient-overlap check: PASSED"
)


# ============================================================
# 7. HELPER:
#    READ DICOM PIXELS
# ============================================================

def load_dicom_pixels(file_path):
    """
    Read a DICOM mammogram and return:
        pixel_array
        metadata dictionary
    """

    ds = pydicom.dcmread(
        str(file_path),
        force=True
    )

    pixel_array = ds.pixel_array

    # Convert to floating point before processing
    pixel_array = pixel_array.astype(
        np.float32
    )

    # Apply modality LUT if available.
    # This respects rescale slope/intercept where present.
    if apply_modality_lut is not None:
        try:
            pixel_array = apply_modality_lut(
                pixel_array,
                ds
            ).astype(
                np.float32
            )
        except Exception:
            pass

    metadata = {
        "rows_original":
            getattr(ds, "Rows", None),

        "columns_original":
            getattr(ds, "Columns", None),

        "photometric_interpretation":
            str(
                getattr(
                    ds,
                    "PhotometricInterpretation",
                    ""
                )
            ),

        "image_laterality":
            str(
                getattr(
                    ds,
                    "ImageLaterality",
                    ""
                )
            ),

        "bits_stored":
            getattr(
                ds,
                "BitsStored",
                None
            )
    }

    return pixel_array, metadata


# ============================================================
# 8. ROBUST INTENSITY NORMALISATION
# ============================================================

def normalize_mammogram(
    image,
    low_percentile=0.5,
    high_percentile=99.5
):
    """
    Convert mammogram pixels into uint8 [0,255].

    Only finite pixels are considered when estimating
    the percentile range.
    """

    image = image.astype(
        np.float32
    )

    finite_mask = np.isfinite(
        image
    )

    if not finite_mask.any():
        raise ValueError(
            "Image contains no finite pixel values."
        )

    valid_pixels = image[
        finite_mask
    ]

    low = np.percentile(
        valid_pixels,
        low_percentile
    )

    high = np.percentile(
        valid_pixels,
        high_percentile
    )

    if high <= low:
        low = np.min(
            valid_pixels
        )

        high = np.max(
            valid_pixels
        )

    if high <= low:
        raise ValueError(
            "Image has no usable intensity range."
        )

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        image - low
    ) / (
        high - low
    )

    image = (
        image * 255.0
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(
        np.uint8
    )

    return image, low, high


# ============================================================
# 9. ASPECT-RATIO PRESERVING RESIZE + PAD
# ============================================================

def resize_and_pad(
    image_array,
    output_size=512
):
    """
    Preserve aspect ratio, resize the mammogram,
    then centre it on a black square canvas.

    This avoids geometric distortion caused by directly
    stretching 2294x1914 images into a square.
    """

    image = Image.fromarray(
        image_array,
        mode="L"
    )

    original_width, original_height = (
        image.size
    )

    scale = min(
        output_size / original_width,
        output_size / original_height
    )

    new_width = max(
        1,
        int(
            round(
                original_width * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                original_height * scale
            )
        )
    )

    resized = image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (
            output_size,
            output_size
        ),
        color=0
    )

    left = (
        output_size
        - new_width
    ) // 2

    top = (
        output_size
        - new_height
    ) // 2

    canvas.paste(
        resized,
        (
            left,
            top
        )
    )

    return (
        canvas,
        new_width,
        new_height
    )


# ============================================================
# 10. PROCESS ONE DICOM IMAGE
# ============================================================

def process_image(
    raw_path,
    output_path
):

    raw_path = Path(
        raw_path
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw DICOM not found: "
            f"{raw_path}"
        )

    # Resume-friendly behaviour
    if (
        output_path.exists()
        and not OVERWRITE
    ):
        return {
            "status": "already_exists",
            "low_intensity": np.nan,
            "high_intensity": np.nan,
            "rows_original": np.nan,
            "columns_original": np.nan,
            "photometric_interpretation": "",
            "image_laterality": "",
            "bits_stored": np.nan,
            "resized_width": np.nan,
            "resized_height": np.nan
        }

    pixel_array, metadata = (
        load_dicom_pixels(
            raw_path
        )
    )

    normalized, low, high = (
        normalize_mammogram(
            pixel_array,
            LOW_PERCENTILE,
            HIGH_PERCENTILE
        )
    )

    processed_image, new_width, new_height = (
        resize_and_pad(
            normalized,
            OUTPUT_SIZE
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_image.save(
        output_path,
        format="PNG",
        optimize=True
    )

    return {
        "status": "processed",

        "low_intensity":
            float(low),

        "high_intensity":
            float(high),

        "rows_original":
            metadata[
                "rows_original"
            ],

        "columns_original":
            metadata[
                "columns_original"
            ],

        "photometric_interpretation":
            metadata[
                "photometric_interpretation"
            ],

        "image_laterality":
            metadata[
                "image_laterality"
            ],

        "bits_stored":
            metadata[
                "bits_stored"
            ],

        "resized_width":
            new_width,

        "resized_height":
            new_height
    }


# ============================================================
# 11. PROCESS ALL BREASTS
# ============================================================

image_records = []

failures = []

total_images_expected = (
    len(df) * 2
)

processed_counter = 0

print(
    f"\nExpected images: "
    f"{total_images_expected:,}"
)

print(
    "\nStarting image preprocessing..."
)


for row_index, row in df.iterrows():

    patient_breast_id = str(
        row[
            "patient_breast_id"
        ]
    )

    patient_id = str(
        row[
            "patient_id"
        ]
    )

    breast_side = str(
        row[
            "breast_side"
        ]
    )

    split = str(
        row[
            "split"
        ]
    )

    target = int(
        row[
            "target"
        ]
    )

    classification = (
        row.get(
            "classification",
            ""
        )
    )

    # Each breast has two images
    image_pairs = [
        (
            1,
            row["image_file_1"]
        ),
        (
            2,
            row["image_file_2"]
        )
    ]

    for view_number, raw_file in image_pairs:

        processed_counter += 1

        output_directory = (
            OUTPUT_ROOT
            / split
            / patient_breast_id
        )

        output_file = (
            output_directory
            / f"view_{view_number}.png"
        )

        try:

            result = process_image(
                raw_file,
                output_file
            )

            record = {
                "patient_breast_id":
                    patient_breast_id,

                "patient_id":
                    patient_id,

                "breast_side":
                    breast_side,

                "target":
                    target,

                "classification":
                    classification,

                "split":
                    split,

                "view_number":
                    view_number,

                "raw_dicom_path":
                    str(raw_file),

                "processed_image_path":
                    str(output_file),

                **result
            }

            image_records.append(
                record
            )

        except Exception as error:

            failures.append({
                "patient_breast_id":
                    patient_breast_id,

                "patient_id":
                    patient_id,

                "breast_side":
                    breast_side,

                "split":
                    split,

                "target":
                    target,

                "view_number":
                    view_number,

                "raw_dicom_path":
                    str(raw_file),

                "error":
                    str(error)
            })

    if (
        row_index + 1
    ) % 100 == 0:

        print(
            f"Processed breasts: "
            f"{row_index + 1:,}"
            f" / {len(df):,}"
        )


# ============================================================
# 12. CREATE IMAGE-LEVEL MANIFEST
# ============================================================

image_manifest = pd.DataFrame(
    image_records
)

IMAGE_MANIFEST_FILE = (
    OUTPUT_ROOT
    / "image_level_manifest.csv"
)

image_manifest.to_csv(
    IMAGE_MANIFEST_FILE,
    index=False
)


# ============================================================
# 13. SAVE FAILURES
# ============================================================

failure_df = pd.DataFrame(
    failures
)

FAILURE_FILE = (
    RESULTS_DIR
    / "image_preprocessing_failures.csv"
)

failure_df.to_csv(
    FAILURE_FILE,
    index=False
)


# ============================================================
# 14. CREATE BREAST-LEVEL MANIFEST
# ============================================================

if len(image_manifest) > 0:

    view1 = (
        image_manifest[
            image_manifest[
                "view_number"
            ] == 1
        ][
            [
                "patient_breast_id",
                "processed_image_path"
            ]
        ]
        .rename(
            columns={
                "processed_image_path":
                    "processed_image_1"
            }
        )
    )

    view2 = (
        image_manifest[
            image_manifest[
                "view_number"
            ] == 2
        ][
            [
                "patient_breast_id",
                "processed_image_path"
            ]
        ]
        .rename(
            columns={
                "processed_image_path":
                    "processed_image_2"
            }
        )
    )

    breast_manifest = (
        df.merge(
            view1,
            on="patient_breast_id",
            how="left",
            validate="one_to_one"
        )
        .merge(
            view2,
            on="patient_breast_id",
            how="left",
            validate="one_to_one"
        )
    )

else:

    breast_manifest = (
        df.copy()
    )

    breast_manifest[
        "processed_image_1"
    ] = pd.NA

    breast_manifest[
        "processed_image_2"
    ] = pd.NA


BREAST_MANIFEST_FILE = (
    OUTPUT_ROOT
    / "breast_level_image_manifest.csv"
)

breast_manifest.to_csv(
    BREAST_MANIFEST_FILE,
    index=False
)


# ============================================================
# 15. VALIDATE OUTPUT COUNTS
# ============================================================

print("\n" + "=" * 78)
print("OUTPUT VALIDATION")
print("=" * 78)

print(
    f"\nExpected images: "
    f"{total_images_expected:,}"
)

print(
    f"Successful manifest records: "
    f"{len(image_manifest):,}"
)

print(
    f"Failed images: "
    f"{len(failure_df):,}"
)


# ============================================================
# 16. CHECK FILE EXISTENCE
# ============================================================

if len(image_manifest) > 0:

    image_manifest[
        "processed_file_exists"
    ] = (
        image_manifest[
            "processed_image_path"
        ]
        .apply(
            lambda x:
            Path(x).exists()
        )
    )

    missing_processed_files = (
        ~image_manifest[
            "processed_file_exists"
        ]
    ).sum()

else:

    missing_processed_files = (
        total_images_expected
    )


print(
    f"Missing processed PNG files: "
    f"{missing_processed_files:,}"
)


# ============================================================
# 17. SPLIT SUMMARY
# ============================================================

if len(image_manifest) > 0:

    split_summary = (
        image_manifest
        .groupby(
            [
                "split",
                "target"
            ]
        )
        .size()
        .reset_index(
            name="image_count"
        )
    )

    print(
        "\nProcessed image distribution "
        "by split / target:"
    )

    print(
        split_summary.to_string(
            index=False
        )
    )

    split_summary.to_csv(
        RESULTS_DIR
        / "processed_image_split_summary.csv",
        index=False
    )


# ============================================================
# 18. BREAST-LEVEL COMPLETENESS
# ============================================================

complete_breasts = (
    breast_manifest[
        "processed_image_1"
    ].notna()
    &
    breast_manifest[
        "processed_image_2"
    ].notna()
)

print(
    "\nComplete breasts with both "
    "processed images:",
    int(
        complete_breasts.sum()
    )
)

print(
    "Incomplete breasts:",
    int(
        (
            ~complete_breasts
        ).sum()
    )
)


# ============================================================
# 19. CHECK LATERALITY CONSISTENCY
# ============================================================

if len(image_manifest) > 0:

    laterality_matches = (
        image_manifest[
            "breast_side"
        ].astype(str)
        ==
        image_manifest[
            "image_laterality"
        ].astype(str)
    )

    # Ignore blank metadata from resume rows if applicable
    valid_laterality = (
        image_manifest[
            "image_laterality"
        ]
        .astype(str)
        .str.strip()
        .ne("")
    )

    laterality_check = (
        laterality_matches[
            valid_laterality
        ]
    )

    if len(
        laterality_check
    ) > 0:

        print(
            "\nClinical ↔ DICOM laterality "
            "agreement:"
        )

        print(
            f"{laterality_check.mean() * 100:.2f}%"
        )


# ============================================================
# 20. IMAGE QUALITY SUMMARY
# ============================================================

if len(image_manifest) > 0:

    quality_summary = pd.DataFrame({
        "measure": [
            "Expected images",
            "Manifest image records",
            "Failed images",
            "Missing PNG files",
            "Complete breast records",
            "Incomplete breast records",
            "Output image size"
        ],

        "value": [
            total_images_expected,
            len(image_manifest),
            len(failure_df),
            missing_processed_files,
            int(
                complete_breasts.sum()
            ),
            int(
                (
                    ~complete_breasts
                ).sum()
            ),
            f"{OUTPUT_SIZE}x{OUTPUT_SIZE}"
        ]
    })

    quality_summary.to_csv(
        RESULTS_DIR
        / "image_preprocessing_summary.csv",
        index=False
    )


# ============================================================
# 21. SAVE FINAL UPDATED IMAGE MANIFEST
# ============================================================

image_manifest.to_csv(
    IMAGE_MANIFEST_FILE,
    index=False
)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 78)
print("MAMMOGRAPHY IMAGE PREPARATION COMPLETE")
print("=" * 78)

print(
    "\nProcessed image directory:"
)

print(
    OUTPUT_ROOT
)

print(
    "\nGenerated:"
)

print(
    "- image_level_manifest.csv"
)

print(
    "- breast_level_image_manifest.csv"
)

print(
    "- image_preprocessing_summary.csv"
)

print(
    "- processed_image_split_summary.csv"
)

print(
    "- image_preprocessing_failures.csv"
)

print(
    "\nOriginal DICOM files were NOT modified."
)

print(
    "\nNEXT STEP:"
)

print(
    "Perform visual quality control on a sample "
    "of processed mammograms before CNN training."
)