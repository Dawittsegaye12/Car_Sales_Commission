import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "car_sales_project"

list_of_files = [
    # Core project structure
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_transformation.py",  # Data ingestion script
    f"src/{project_name}/components/feature_extraction.py",  # Image feature extraction script
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/custome_exception.py",  # Text generation model script
    f"src/{project_name}/components/logging.py",
    f"src/{project_name}/components/custome_exception.py",


    "requirements.txt",  # List of Python dependencies
    "setup.py",  # Setup script for the project
    "templates/index.html", # HTML template for any web UI

    "EDA.ipynb", #exploratory data analysis
    "APP.py", #application using flask
]

# Create the directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
