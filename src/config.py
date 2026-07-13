"""
Project Configuration File
Author : Vishal
Project: App User Behavior Segmentation using Unsupervised Machine Learning
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "app_user_behavior_dataset.csv"

RANDOM_STATE = 42

N_CLUSTERS = 4

FIGURE_DPI = 300

OUTPUT_PATH = PROJECT_ROOT / "reports" / "figures"