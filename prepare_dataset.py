#!/usr/bin/env python3
"""
CompHRDoc Dataset Preparation Script

This script copies document images from the HRDH source folder to the datasets folder
according to the CompHRDoc structure. It organizes images into train and test splits
based on the JSON files in the HRDH/train and HRDH/test directories.

Dataset Structure:
- Source: HRDH/images/{document_id}/{page_number}.png
- Target: datasets/Comp-HRDoc/HRDH_MSRA_POD_TRAIN/Images/{document_id}_{page_number}.png
         datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/Images/{document_id}_{page_number}.png

Usage:
    python prepare_dataset.py [--source_dir HRDH] [--target_dir datasets/Comp-HRDoc]
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Set, List
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_document_ids_from_json_files(json_dir: Path) -> Set[str]:
    """
    Extract document IDs from JSON files in the given directory.
    
    Args:
        json_dir: Path to directory containing JSON files
        
    Returns:
        Set of document IDs (without .json extension)
    """
    document_ids = set()
    
    if not json_dir.exists():
        logger.warning(f"JSON directory does not exist: {json_dir}")
        return document_ids
    
    for json_file in json_dir.glob("*.json"):
        if json_file.name != ".DS_Store":  # Skip system files
            document_id = json_file.stem  # Remove .json extension
            document_ids.add(document_id)
            logger.debug(f"Found document ID: {document_id}")
    
    logger.info(f"Found {len(document_ids)} document IDs in {json_dir}")
    return document_ids


def copy_images_for_documents(
    source_images_dir: Path,
    target_images_dir: Path,
    document_ids: Set[str],
    split_name: str
) -> int:
    """
    Copy images for specified documents from source to target directory.
    
    Args:
        source_images_dir: Path to HRDH/images directory
        target_images_dir: Path to target Images directory
        document_ids: Set of document IDs to copy
        split_name: Name of the split (train/test) for logging
        
    Returns:
        Number of images copied
    """
    images_copied = 0
    
    # Create target directory if it doesn't exist
    target_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target directory: {target_images_dir}")
    
    for doc_id in document_ids:
        source_doc_dir = source_images_dir / doc_id
        
        if not source_doc_dir.exists():
            logger.warning(f"Source directory for document {doc_id} does not exist: {source_doc_dir}")
            continue
        
        if not source_doc_dir.is_dir():
            logger.warning(f"Source path for document {doc_id} is not a directory: {source_doc_dir}")
            continue
        
        # Get all PNG files in the document directory
        png_files = list(source_doc_dir.glob("*.png"))
        
        if not png_files:
            logger.warning(f"No PNG files found for document {doc_id}")
            continue
        
        logger.info(f"Processing document {doc_id}: found {len(png_files)} images")
        
        for png_file in png_files:
            # Create target filename: {document_id}_{page_number}.png
            page_number = png_file.stem  # Get filename without extension
            target_filename = f"{doc_id}_{page_number}.png"
            target_path = target_images_dir / target_filename
            
            try:
                # Copy the file
                shutil.copy2(png_file, target_path)
                images_copied += 1
                logger.debug(f"Copied: {png_file} -> {target_path}")
                
            except Exception as e:
                logger.error(f"Failed to copy {png_file} to {target_path}: {e}")
    
    logger.info(f"Completed {split_name} split: copied {images_copied} images for {len(document_ids)} documents")
    return images_copied


def verify_target_structure(target_base_dir: Path) -> bool:
    """
    Verify that the target dataset structure exists and has required files.
    
    Args:
        target_base_dir: Path to datasets/Comp-HRDoc directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = [
        "HRDH_MSRA_POD_TRAIN",
        "HRDH_MSRA_POD_TEST"
    ]
    
    required_files = [
        "HRDH_MSRA_POD_TRAIN/hdsa_train.json",
        "HRDH_MSRA_POD_TRAIN/coco_train.json",
        "HRDH_MSRA_POD_TEST/hdsa_test.json",
        "HRDH_MSRA_POD_TEST/coco_test.json",
        "HRDH_MSRA_POD_TEST/test_eval",
        "HRDH_MSRA_POD_TEST/test_eval_toc"
    ]
    
    # Check directories
    for req_dir in required_dirs:
        dir_path = target_base_dir / req_dir
        if not dir_path.exists():
            logger.error(f"Required directory missing: {dir_path}")
            return False
    
    # Check files and directories
    for req_file in required_files:
        file_path = target_base_dir / req_file
        if not file_path.exists():
            logger.warning(f"Required file/directory missing: {file_path}")
    
    logger.info("Target dataset structure verification completed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CompHRDoc dataset by copying images from HRDH source to dataset structure"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="HRDH",
        help="Path to source HRDH directory (default: HRDH)"
    )
    parser.add_argument(
        "--target_dir", 
        type=str,
        default="datasets/Comp-HRDoc",
        help="Path to target Comp-HRDoc directory (default: datasets/Comp-HRDoc)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert to Path objects
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    
    logger.info("="*60)
    logger.info("CompHRDoc Dataset Preparation")
    logger.info("="*60)
    logger.info(f"Source directory: {source_dir.absolute()}")
    logger.info(f"Target directory: {target_dir.absolute()}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
    
    # Verify source structure
    source_images_dir = source_dir / "images"
    source_train_dir = source_dir / "train"
    source_test_dir = source_dir / "test"
    
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return 1
    
    if not source_images_dir.exists():
        logger.error(f"Source images directory does not exist: {source_images_dir}")
        return 1
    
    if not source_train_dir.exists():
        logger.error(f"Source train directory does not exist: {source_train_dir}")
        return 1
    
    if not source_test_dir.exists():
        logger.error(f"Source test directory does not exist: {source_test_dir}")
        return 1
    
    # Verify target structure
    if not verify_target_structure(target_dir):
        logger.error("Target dataset structure is incomplete")
        return 1
    
    # Get document IDs for train and test splits
    logger.info("Analyzing train/test splits...")
    train_doc_ids = get_document_ids_from_json_files(source_train_dir)
    test_doc_ids = get_document_ids_from_json_files(source_test_dir)
    
    if not train_doc_ids:
        logger.error("No training documents found")
        return 1
    
    if not test_doc_ids:
        logger.error("No test documents found")
        return 1
    
    # Check for overlap
    overlap = train_doc_ids.intersection(test_doc_ids)
    if overlap:
        logger.warning(f"Found {len(overlap)} documents in both train and test splits: {list(overlap)[:5]}...")
    
    logger.info(f"Train split: {len(train_doc_ids)} documents")
    logger.info(f"Test split: {len(test_doc_ids)} documents")
    logger.info(f"Total unique documents: {len(train_doc_ids.union(test_doc_ids))}")
    
    if args.dry_run:
        logger.info("Dry run completed - no files were copied")
        return 0
    
    # Copy images for training set
    logger.info("="*40)
    logger.info("Copying TRAINING images...")
    logger.info("="*40)
    train_target_dir = target_dir / "HRDH_MSRA_POD_TRAIN" / "Images"
    train_images_copied = copy_images_for_documents(
        source_images_dir, train_target_dir, train_doc_ids, "training"
    )
    
    # Copy images for test set
    logger.info("="*40)
    logger.info("Copying TEST images...")
    logger.info("="*40)
    test_target_dir = target_dir / "HRDH_MSRA_POD_TEST" / "Images"
    test_images_copied = copy_images_for_documents(
        source_images_dir, test_target_dir, test_doc_ids, "test"
    )
    
    # Summary
    total_images = train_images_copied + test_images_copied
    logger.info("="*60)
    logger.info("DATASET PREPARATION COMPLETED")
    logger.info("="*60)
    logger.info(f"Training images copied: {train_images_copied}")
    logger.info(f"Test images copied: {test_images_copied}")
    logger.info(f"Total images copied: {total_images}")
    logger.info(f"Training documents: {len(train_doc_ids)}")
    logger.info(f"Test documents: {len(test_doc_ids)}")
    
    if total_images == 0:
        logger.warning("No images were copied! Please check source directory structure.")
        return 1
    
    logger.info("Dataset preparation completed successfully!")
    logger.info(f"Log file saved to: dataset_preparation.log")
    
    return 0


if __name__ == "__main__":
    exit(main())
