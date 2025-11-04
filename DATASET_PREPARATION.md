# Dataset Preparation Guide

This document explains how to use the `prepare_dataset.py` script to set up the CompHRDoc dataset.

## Overview

The `prepare_dataset.py` script copies document images from the HRDH source folder to the proper CompHRDoc dataset structure. It organizes images into training and test splits based on the JSON files provided in the dataset.

## Prerequisites

1. **Download the HRDoc-Hard dataset** from [HRDoc-Hard dataset repository](https://github.com/jfma-USTC/HRDoc)
2. **Extract the images** to the `HRDH/images/` directory
3. **Ensure the CompHRDoc annotations** are already in place in `datasets/Comp-HRDoc/`

## Directory Structure

### Source Structure (HRDH folder):

```
HRDH/
├── images/
│   ├── 1401.3699/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── 1401.6399/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   └── ...
├── train/
│   ├── 1401.6399.json
│   ├── 1401.8087.json
│   └── ...
└── test/
    ├── 1401.3699.json
    ├── 1402.2741.json
    └── ...
```

### Target Structure (datasets folder):

```
datasets/Comp-HRDoc/
├── HRDH_MSRA_POD_TRAIN/
│   ├── Images/
│   │   ├── 1401.6399_0.png
│   │   ├── 1401.6399_1.png
│   │   └── ...
│   ├── hdsa_train.json
│   ├── coco_train.json
│   └── ...
└── HRDH_MSRA_POD_TEST/
    ├── Images/
    │   ├── 1401.3699_0.png
    │   ├── 1401.3699_1.png
    │   └── ...
    ├── hdsa_test.json
    ├── coco_test.json
    ├── test_eval/
    ├── test_eval_toc/
    └── ...
```

## Usage

### Basic Usage

```bash
python prepare_dataset.py
```

### Custom Directories

```bash
python prepare_dataset.py --source_dir /path/to/HRDH --target_dir /path/to/datasets/Comp-HRDoc
```

### Dry Run (preview what will be copied)

```bash
python prepare_dataset.py --dry_run
```

### Verbose Output

```bash
python prepare_dataset.py --verbose
```

### All Options

```bash
python prepare_dataset.py --source_dir HRDH --target_dir datasets/Comp-HRDoc --dry_run --verbose
```

## Command Line Options

| Option         | Description                                        | Default               |
| -------------- | -------------------------------------------------- | --------------------- |
| `--source_dir` | Path to source HRDH directory                      | `HRDH`                |
| `--target_dir` | Path to target Comp-HRDoc directory                | `datasets/Comp-HRDoc` |
| `--dry_run`    | Show what would be copied without actually copying | False                 |
| `--verbose`    | Enable verbose logging                             | False                 |

## What the Script Does

1. **Validates directories**: Checks that source and target directories exist and have the expected structure
2. **Analyzes splits**: Reads JSON files from `HRDH/train/` and `HRDH/test/` to determine which documents belong to which split
3. **Copies images**: For each document in the train/test splits:
   - Finds the document's image folder in `HRDH/images/{document_id}/`
   - Copies all PNG files to the target directory with renamed format `{document_id}_{page_number}.png`
4. **Logs progress**: Provides detailed logging of the copy process

## Expected Output

After running the script successfully, you should see:

```
CompHRDoc Dataset Preparation
============================================================
Source directory: /path/to/HRDH
Target directory: /path/to/datasets/Comp-HRDoc

Train split: 1000 documents
Test split: 500 documents
Total unique documents: 1500

========================================
Copying TRAINING images...
========================================
Completed training split: copied XXXX images for 1000 documents

========================================
Copying TEST images...
========================================
Completed test split: copied XXXX images for 500 documents

============================================================
DATASET PREPARATION COMPLETED
============================================================
Training images copied: XXXX
Test images copied: XXXX
Total images copied: XXXX
Training documents: 1000
Test documents: 500

Dataset preparation completed successfully!
```

## Troubleshooting

### Common Issues

1. **"Source directory does not exist"**

   - Make sure you've downloaded and extracted the HRDoc-Hard dataset
   - Check that the HRDH folder is in the correct location

2. **"No training/test documents found"**

   - Verify that the HRDH/train/ and HRDH/test/ directories contain JSON files
   - Check that the CompHRDoc.zip has been properly extracted

3. **"No PNG files found for document"**

   - Some documents may not have associated images
   - This is normal and will be logged as a warning

4. **Permission errors**
   - Make sure you have write permissions to the target directory
   - Try running with administrator privileges if needed

### Log File

The script creates a detailed log file named `dataset_preparation.log` in the current directory. Check this file for detailed information about the copying process and any errors encountered.

## Verification

After running the script, you can verify the setup by checking:

1. **Image counts**: The target Images folders should contain thousands of images
2. **File naming**: Images should be named like `1401.3699_0.png`, `1401.3699_1.png`, etc.
3. **Split integrity**: Training and test images should be in separate folders

## Next Steps

After preparing the dataset:

1. **Verify the installation** of all required dependencies (PyTorch, detectron2, detrex, etc.)
2. **Test the model** by running inference or training scripts
3. **Check the evaluation tools** in the `evaluation/` directory

For training the UniHDSA model, refer to the configuration files in `UniHDSA/configs/`.
