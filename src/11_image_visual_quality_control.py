# ============================================================
# TOMPEI-CMMD
# IMAGE VISUAL QUALITY CONTROL
#
# Purpose:
#   - Visually inspect processed mammograms
#   - Sample across Train / Validation / Test
#   - Include malignant and non-malignant examples
#   - Check for blank / overly dark / overly bright images
#   - Save QC figures and statistics
#
# IMPORTANT:
#   Processed images are READ ONLY.
#   No images are modified.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\LAPTOP CLINIC\Downloads\Dissertation"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "03_Data_Processed"
    / "mammography_512"
)

IMAGE_MANIFEST = (
    IMAGE_ROOT
    / "image_level_manifest.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "05_Results"
    / "image_preprocessing"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# 2 samples per split/class gives:
# 3 splits x 2 classes x 2 images = 12 images
SAMPLES_PER_GROUP = 2


# ============================================================
# 3. LOAD MANIFEST
# ============================================================

print("=" * 78)
print("TOMPEI-CMMD IMAGE VISUAL QUALITY CONTROL")
print("=" * 78)

if not IMAGE_MANIFEST.exists():
    raise FileNotFoundError(
        f"Image manifest not found:\n{IMAGE_MANIFEST}"
    )

df = pd.read_csv(
    IMAGE_MANIFEST
)

print(
    f"\nImage records loaded: "
    f"{len(df):,}"
)

print(
    f"Unique breasts: "
    f"{df['patient_breast_id'].nunique():,}"
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
    "classification",
    "split",
    "view_number",
    "processed_image_path"
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
# 5. CHECK IMAGE FILE EXISTENCE
# ============================================================

df["file_exists"] = (
    df["processed_image_path"]
    .apply(
        lambda x:
        Path(x).exists()
    )
)

missing_files = (
    ~df["file_exists"]
).sum()

print(
    f"\nMissing processed image files: "
    f"{missing_files:,}"
)

if missing_files > 0:

    missing_df = df[
        ~df["file_exists"]
    ].copy()

    missing_df.to_csv(
        RESULTS_DIR
        / "qc_missing_processed_images.csv",
        index=False
    )

    raise FileNotFoundError(
        "Some processed images are missing."
    )


# ============================================================
# 6. IMAGE STATISTICS
# ============================================================

print(
    "\nCalculating image statistics..."
)

statistics = []

for index, row in df.iterrows():

    image_path = Path(
        row[
            "processed_image_path"
        ]
    )

    image = Image.open(
        image_path
    ).convert(
        "L"
    )

    array = np.array(
        image,
        dtype=np.float32
    )

    mean_intensity = (
        float(
            array.mean()
        )
    )

    std_intensity = (
        float(
            array.std()
        )
    )

    min_intensity = (
        float(
            array.min()
        )
    )

    max_intensity = (
        float(
            array.max()
        )
    )

    black_fraction = (
        float(
            (array <= 5).mean()
        )
    )

    bright_fraction = (
        float(
            (array >= 250).mean()
        )
    )

    statistics.append({

        "patient_breast_id":
            row[
                "patient_breast_id"
            ],

        "patient_id":
            row[
                "patient_id"
            ],

        "breast_side":
            row[
                "breast_side"
            ],

        "target":
            row[
                "target"
            ],

        "classification":
            row[
                "classification"
            ],

        "split":
            row[
                "split"
            ],

        "view_number":
            row[
                "view_number"
            ],

        "processed_image_path":
            str(
                image_path
            ),

        "mean_intensity":
            mean_intensity,

        "std_intensity":
            std_intensity,

        "min_intensity":
            min_intensity,

        "max_intensity":
            max_intensity,

        "black_fraction":
            black_fraction,

        "bright_fraction":
            bright_fraction,

        "width":
            image.width,

        "height":
            image.height
    })


stats_df = pd.DataFrame(
    statistics
)


# ============================================================
# 7. SAVE IMAGE STATISTICS
# ============================================================

stats_df.to_csv(
    RESULTS_DIR
    / "image_qc_statistics.csv",
    index=False
)


# ============================================================
# 8. GLOBAL QC SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("IMAGE STATISTICS SUMMARY")
print("=" * 78)

summary_columns = [
    "mean_intensity",
    "std_intensity",
    "black_fraction",
    "bright_fraction"
]

print(
    stats_df[
        summary_columns
    ]
    .describe()
    .to_string()
)


# ============================================================
# 9. CHECK IMAGE DIMENSIONS
# ============================================================

dimension_summary = (
    stats_df.groupby(
        [
            "width",
            "height"
        ]
    )
    .size()
    .reset_index(
        name="image_count"
    )
)

print("\nImage dimensions:")

print(
    dimension_summary
    .to_string(
        index=False
    )
)

dimension_summary.to_csv(
    RESULTS_DIR
    / "image_qc_dimension_summary.csv",
    index=False
)


# ============================================================
# 10. FLAG POTENTIALLY SUSPICIOUS IMAGES
# ============================================================

# These thresholds are intentionally conservative.
# They are flags for manual review, not automatic deletion.

stats_df[
    "flag_low_variance"
] = (
    stats_df[
        "std_intensity"
    ] < 5
)

stats_df[
    "flag_extremely_dark"
] = (
    stats_df[
        "mean_intensity"
    ] < 5
)

stats_df[
    "flag_extremely_bright"
] = (
    stats_df[
        "mean_intensity"
    ] > 250
)

stats_df[
    "flag_mostly_black"
] = (
    stats_df[
        "black_fraction"
    ] > 0.95
)

stats_df[
    "qc_flag"
] = (
    stats_df[
        [
            "flag_low_variance",
            "flag_extremely_dark",
            "flag_extremely_bright",
            "flag_mostly_black"
        ]
    ]
    .any(
        axis=1
    )
)

flagged = stats_df[
    stats_df[
        "qc_flag"
    ]
].copy()

flagged.to_csv(
    RESULTS_DIR
    / "image_qc_flagged_images.csv",
    index=False
)

print(
    "\nPotentially suspicious images flagged:",
    len(flagged)
)


# ============================================================
# 11. SAMPLE ONE IMAGE PER BREAST
#    FOR MAIN QC MONTAGE
# ============================================================

# Use view 1 for a clean balanced comparison
view1_df = df[
    df[
        "view_number"
    ] == 1
].copy()


sample_groups = []

for split in [
    "train",
    "validation",
    "test"
]:

    for target in [
        0,
        1
    ]:

        group = view1_df[
            (
                view1_df[
                    "split"
                ] == split
            )
            &
            (
                view1_df[
                    "target"
                ] == target
            )
        ]

        if len(group) < SAMPLES_PER_GROUP:

            raise ValueError(
                f"Not enough samples for "
                f"{split}, target={target}"
            )

        sampled = group.sample(
            n=SAMPLES_PER_GROUP,
            random_state=(
                RANDOM_STATE
                + target
                + len(split)
            )
        )

        sample_groups.append(
            sampled
        )


sample_df = pd.concat(
    sample_groups,
    ignore_index=True
)


# ============================================================
# 12. MAIN QC MONTAGE
# ============================================================

rows = 3
columns = 4

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(12, 11)
)

axes = axes.flatten()


for ax, (_, row) in zip(
    axes,
    sample_df.iterrows()
):

    image = Image.open(
        row[
            "processed_image_path"
        ]
    ).convert(
        "L"
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    class_name = (
        "Malignant"
        if row[
            "target"
        ] == 1
        else "Non-malignant"
    )

    ax.set_title(
        f"{row['split'].title()} | "
        f"{class_name}\n"
        f"{row['patient_breast_id']} | "
        f"View {row['view_number']}",
        fontsize=9
    )

    ax.axis(
        "off"
    )


plt.suptitle(
    "Processed Mammography Visual Quality Control",
    fontsize=15
)

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)

plt.savefig(
    FIGURE_DIR
    / "01_processed_mammography_qc_montage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. TWO-VIEW BREAST QC
# ============================================================

# Randomly select 6 breasts:
# 3 non-malignant + 3 malignant

breast_targets = (
    df[
        [
            "patient_breast_id",
            "target",
            "split"
        ]
    ]
    .drop_duplicates(
        subset=[
            "patient_breast_id"
        ]
    )
)


selected_breasts = []

for target in [
    0,
    1
]:

    group = breast_targets[
        breast_targets[
            "target"
        ] == target
    ]

    selected = group.sample(
        n=3,
        random_state=(
            RANDOM_STATE
            + target
        )
    )

    selected_breasts.extend(
        selected[
            "patient_breast_id"
        ].tolist()
    )


two_view_sample = df[
    df[
        "patient_breast_id"
    ].isin(
        selected_breasts
    )
].copy()

two_view_sample = (
    two_view_sample
    .sort_values(
        [
            "patient_breast_id",
            "view_number"
        ]
    )
)


fig, axes = plt.subplots(
    6,
    2,
    figsize=(8, 20)
)


for row_index, breast_id in enumerate(
    selected_breasts
):

    breast_images = (
        two_view_sample[
            two_view_sample[
                "patient_breast_id"
            ] == breast_id
        ]
        .sort_values(
            "view_number"
        )
    )

    for column_index, (
        _,
        image_row
    ) in enumerate(
        breast_images.iterrows()
    ):

        image = Image.open(
            image_row[
                "processed_image_path"
            ]
        ).convert(
            "L"
        )

        axes[
            row_index,
            column_index
        ].imshow(
            image,
            cmap="gray"
        )

        class_name = (
            "Malignant"
            if image_row[
                "target"
            ] == 1
            else "Non-malignant"
        )

        axes[
            row_index,
            column_index
        ].set_title(
            f"{breast_id} | "
            f"{class_name} | "
            f"View "
            f"{image_row['view_number']}",
            fontsize=9
        )

        axes[
            row_index,
            column_index
        ].axis(
            "off"
        )


plt.suptitle(
    "Two-View Mammography Quality Control",
    fontsize=15
)

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.98
    ]
)

plt.savefig(
    FIGURE_DIR
    / "02_two_view_breast_qc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. INTENSITY HISTOGRAM
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    stats_df[
        "mean_intensity"
    ],
    bins=40
)

plt.xlabel(
    "Mean Pixel Intensity"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Distribution of Processed Mammogram Mean Intensity"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "03_mean_intensity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. STANDARD DEVIATION HISTOGRAM
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    stats_df[
        "std_intensity"
    ],
    bins=40
)

plt.xlabel(
    "Pixel Intensity Standard Deviation"
)

plt.ylabel(
    "Number of Images"
)

plt.title(
    "Processed Mammogram Intensity Variation"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "04_intensity_std_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. QC SUMMARY
# ============================================================

qc_summary = pd.DataFrame({

    "measure": [
        "Total processed images",
        "Unique breasts",
        "Unique patients",
        "Missing image files",
        "Images flagged for review",
        "Expected width",
        "Expected height"
    ],

    "value": [
        len(df),
        df[
            "patient_breast_id"
        ].nunique(),
        df[
            "patient_id"
        ].nunique(),
        int(
            missing_files
        ),
        len(
            flagged
        ),
        512,
        512
    ]
})

qc_summary.to_csv(
    RESULTS_DIR
    / "image_visual_qc_summary.csv",
    index=False
)


# ============================================================
# 17. SAVE SAMPLE INFORMATION
# ============================================================

sample_df.to_csv(
    RESULTS_DIR
    / "image_qc_montage_samples.csv",
    index=False
)

two_view_sample.to_csv(
    RESULTS_DIR
    / "image_qc_two_view_samples.csv",
    index=False
)


# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 78)
print("IMAGE VISUAL QUALITY CONTROL COMPLETE")
print("=" * 78)

print(
    "\nGenerated figures:"
)

print(
    "- 01_processed_mammography_qc_montage.png"
)

print(
    "- 02_two_view_breast_qc.png"
)

print(
    "- 03_mean_intensity_distribution.png"
)

print(
    "- 04_intensity_std_distribution.png"
)

print(
    "\nGenerated QC files:"
)

print(
    "- image_qc_statistics.csv"
)

print(
    "- image_qc_flagged_images.csv"
)

print(
    "- image_visual_qc_summary.csv"
)

print(
    "\nProcessed mammograms were NOT modified."
)

print(
    "\nNEXT STEP:"
)

print(
    "Open the two QC montage figures and visually "
    "confirm that breast tissue is clearly visible "
    "before beginning CNN training."
)