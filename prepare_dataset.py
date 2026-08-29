import os
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = "mrl_dataset"
OUTPUT_DIR = "dataset"


# ============================================================
# SOURCE DIRECTORIES
# ============================================================

source_folders = {
    "train_open": os.path.join(
        SOURCE_DIR,
        "train",
        "open_eyes_sample"
    ),

    "train_closed": os.path.join(
        SOURCE_DIR,
        "train",
        "close_eyes_sample"
    ),

    "test_open": os.path.join(
        SOURCE_DIR,
        "test",
        "open_eyes_test"
    ),

    "test_closed": os.path.join(
        SOURCE_DIR,
        "test",
        "close_eyes_test"
    )
}


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

output_folders = {
    "train_open": os.path.join(
        OUTPUT_DIR,
        "train",
        "open"
    ),

    "train_closed": os.path.join(
        OUTPUT_DIR,
        "train",
        "closed"
    ),

    "test_open": os.path.join(
        OUTPUT_DIR,
        "test",
        "open"
    ),

    "test_closed": os.path.join(
        OUTPUT_DIR,
        "test",
        "closed"
    )
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

print("\nCreating output directories...")

for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(source, destination):

    if not os.path.exists(source):

        print("\nERROR: Source folder not found:")
        print(source)

        return 0

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    images = [
        file
        for file in os.listdir(source)
        if file.lower().endswith(valid_extensions)
    ]

    print("\nSource:")
    print(source)

    print("Images:", len(images))

    for index, image in enumerate(images, start=1):

        source_path = os.path.join(
            source,
            image
        )

        new_name = f"{index:06d}_{image}"

        destination_path = os.path.join(
            destination,
            new_name
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        if index % 500 == 0:

            print(
                f"Copied {index}/{len(images)}"
            )

    print("Completed.")

    return len(images)


# ============================================================
# PROCESS DATASET
# ============================================================

train_open_count = copy_images(
    source_folders["train_open"],
    output_folders["train_open"]
)

train_closed_count = copy_images(
    source_folders["train_closed"],
    output_folders["train_closed"]
)

test_open_count = copy_images(
    source_folders["test_open"],
    output_folders["test_open"]
)

test_closed_count = copy_images(
    source_folders["test_closed"],
    output_folders["test_closed"]
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 50)
print("DATASET PREPARATION COMPLETED")
print("=" * 50)

print(
    f"Train Open   : {train_open_count}"
)

print(
    f"Train Closed : {train_closed_count}"
)

print(
    f"Test Open    : {test_open_count}"
)

print(
    f"Test Closed  : {test_closed_count}"
)

print("\nFinal structure:")
print("""
dataset/
│
├── train/
│   ├── open/
│   └── closed/
│
└── test/
    ├── open/
    └── closed/
""")

print("Dataset is ready for CNN training.")