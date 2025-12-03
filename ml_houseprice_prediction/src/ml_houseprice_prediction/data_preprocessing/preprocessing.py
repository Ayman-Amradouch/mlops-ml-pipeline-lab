"""
data_preprocessing.py

A clean, modular data preprocessing script for MLOps pipelines.

This script:
- Loads raw data from CSV
- Cleans columns (trims whitespace, removes duplicates, drops NaNs)
- Saves the cleaned dataset to a standardized datastore path

"""

from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging


# -------------------------------------------------------------------
#  Global Configuration
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASTORE_DIR = PROJECT_ROOT / "datastores" 

OUTPUT_DIR = DATASTORE_DIR / "clean_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = DATASTORE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
#  Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) ,
        logging.FileHandler(f"{LOG_DIR}/data_preprocessing.log", mode="a"),
    ],
)
logger = logging.getLogger("data_preprocessing")

logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

# -------------------------------------------------------------------
#  Functions
# -------------------------------------------------------------------
def load_data(input_data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        input_data_path (Union[str, Path]): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger.info("Loading raw dataset...")
    try:
        input_path = Path(input_data_path)
        logger.info(f"Loading raw data from: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the provided dataset by stripping column names,
    removing duplicates, and dropping missing values.

    Args:
        df (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    logger.info("Cleaning dataset...")
    df.columns = df.columns.str.strip()              # Retirer les espaces autour des noms de colonnes
    df = df.drop_duplicates()                        # Supprimer les doublons
    df = df.dropna()                                 # Supprimer les valeurs manquantes
    logger.info(f"Cleaned dataset shape: {df.shape}") # Log du résultat final
    return df



def save_data(df: pd.DataFrame, output_data_filename: str) -> Path:
    """
    Saves the cleaned DataFrame as a CSV to the standardized data directory.

    Args:
        df (pd.DataFrame): Cleaned dataset to save.
        output_data_filename (str): Name of the output CSV file.

    Returns:
        Path: The path where the cleaned file was saved.
    """
    output_path = OUTPUT_DIR / output_data_filename   # Chemin du fichier final
    df.to_csv(output_path, index=False)               # Enregistrer le CSV sans l’index
    logger.info(f"Saved cleaned dataset to: {output_path}")  # Log
    return output_path




# -------------------------------------------------------------------
#  CLI Interface
# -------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments for the preprocessing step.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Data Preprocessing - Clean raw dataset for MLOps pipeline"
    )

    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to the raw input CSV file (e.g., ../datastores/raw_data/data.csv).",
    )

    parser.add_argument(
        "--output_data_filename",
        type=str,
        required=True,
        default="clean_housing.csv",
        help="Output filename (will be stored under ../datastores/clean_data/)",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
#  Main Entry Point
# -------------------------------------------------------------------
def main() -> None:
    """
    Main function for CLI execution.
    Loads, cleans, and saves data in one reproducible pipeline step.
    """
    args = parse_arguments()

    df = load_data(args.input_data_path)                   # Charger données brutes
    cleaned_df = clean_data(df)                            # Nettoyer
    save_data(cleaned_df, args.output_data_filename)       # Sauvegarder

    


if __name__ == "__main__":
    main()
