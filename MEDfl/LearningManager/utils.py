#!/usr/bin/env python3

import pkg_resources
import torch
import yaml
from sklearn.metrics import *
from yaml.loader import SafeLoader


from MEDfl.NetManager.database_connector import DatabaseManager

# from scripts.base import *
import json


import pandas as pd
import numpy as np

import os
import configparser

import subprocess
import ast

from sqlalchemy import text


# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load configuration from the config file
yaml_path = os.path.join(current_directory, 'params.yaml')

with open(yaml_path) as g:
    params = yaml.load(g, Loader=SafeLoader)

# global_yaml_path = pkg_resources.resource_filename(__name__, "../../global_params.yaml")
# with open(global_yaml_path) as g:
#     global_params = yaml.load(g, Loader=SafeLoader)


# Default path for the config file
DEFAULT_CONFIG_PATH = 'db_config.ini'


def load_db_config_dep():
    config = os.environ.get('MEDfl_DB_CONFIG')

    if config:
        return ast.literal_eval(config)
    else:
        raise ValueError(f"MEDfl db config not found")

# Function to allow users to set config path programmatically


def set_db_config_dep(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    if (config['sqllite']):
        os.environ['MEDfl_DB_CONFIG'] = str(dict(config['sqllite']))
    else:
        raise ValueError(f"mysql key not found in file '{config_path}'")



def load_db_config():
    """Read a dictionary from an environment variable."""
    obj_str = os.getenv("MEDfl_DB_CONFIG")
    if obj_str is not None:
        return ast.literal_eval(obj_str)
    else:
        raise ValueError(f"Environment variable MEDfl_DB_CONFIG not found")

# Function to allow users to set config path programmatically


def set_db_config(config_path):
    obj = {"database" : config_path}

    """Store a dictionary as a string in an environment variable."""
    obj_str = str(obj)
    os.environ['MEDfl_DB_CONFIG'] = obj_str

    




# Create databas


def create_MEDfl_db():
    script_path = os.path.join(os.path.dirname(
        __file__), 'scripts', 'create_db.sh')
    subprocess.run(['sh', script_path], check=True)


def custom_classification_report(y_true, y_pred_prob):
    """
    Compute custom classification report metrics including accuracy, sensitivity, specificity, precision, NPV,
    F1-score, false positive rate, and true positive rate.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: A dictionary containing custom classification report metrics.
    """
    y_pred = (y_pred_prob).round(
    )  # Round absolute values of predicted probabilities to the nearest integer

    auc = roc_auc_score(y_true, y_pred_prob)  # Calculate AUC

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred , labels=[0, 1]).ravel()

    # Accuracy
    denominator_acc = tp + tn + fp + fn
    acc = (tp + tn) / denominator_acc if denominator_acc != 0 else 0.0

    # Sensitivity/Recall
    denominator_sen = tp + fn
    sen = tp / denominator_sen if denominator_sen != 0 else 0.0

    # Specificity
    denominator_sp = tn + fp
    sp = tn / denominator_sp if denominator_sp != 0 else 0.0

    # PPV/Precision
    denominator_ppv = tp + fp
    ppv = tp / denominator_ppv if denominator_ppv != 0 else 0.0

#     NPV
    denominator_npv = tn + fn
    npv = tn / denominator_npv if denominator_npv != 0 else 0.0

    # F1 Score
    denominator_f1 = sen + ppv
    f1 = 2 * (sen * ppv) / denominator_f1 if denominator_f1 != 0 else 0.0

    # False Positive Rate
    denominator_fpr = fp + tn
    fpr = fp / denominator_fpr if denominator_fpr != 0 else 0.0

    # True Positive Rate
    denominator_tpr = tp + fn
    tpr = tp / denominator_tpr if denominator_tpr != 0 else 0.0

    return {
        "confusion matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "Accuracy": round(acc, 3),
        "Sensitivity/Recall": round(sen, 3),
        "Specificity": round(sp, 3),
        "PPV/Precision": round(ppv, 3),
        "NPV": round(npv, 3),
        "F1-score": round(f1, 3),
        "False positive rate": round(fpr, 3),
        "True positive rate": round(tpr, 3),
        "auc": auc
    }


def test(model, test_loader, device=torch.device("cpu")):
    """
    Evaluate a model using a test loader and return a custom classification report.

    Args:
        model (torch.nn.Module): PyTorch model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device, optional): Device for model evaluation. Default is "cpu".

    Returns:
        dict: A dictionary containing custom classification report metrics.
    """

    model.eval()
    with torch.no_grad():
        X_test, y_test = test_loader.dataset[:][0].to(
            device), test_loader.dataset[:][1].to(device)
        y_hat_prob = torch.squeeze(model(X_test), 1).cpu()

    return custom_classification_report(y_test.cpu().numpy(), y_hat_prob.cpu().numpy())


column_map = {"object": "VARCHAR(255)", "int64": "INT", "float64": "FLOAT"}


def empty_db():
    """
    Empty the database by deleting records from multiple tables and resetting auto-increment counters.

    Returns:
        None
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    # my_eng.execute(text(f"DELETE FROM  {'DataSets'}"))
    my_eng.execute(text(f"DELETE FROM {'Nodes'}"))
    my_eng.execute(text(f"DELETE FROM {'FedDatasets'}"))
    my_eng.execute(text(f"DELETE FROM {'Networks'}"))
    my_eng.execute(text(f"DELETE FROM {'FLsetup'}"))

    my_eng.execute(text(f"DELETE FROM {'FLpipeline'}"))
    my_eng.execute(text(f"ALTER TABLE {'Nodes'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'Networks'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FedDatasets'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FLsetup'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"ALTER TABLE {'FLpipeline'} AUTO_INCREMENT = 1"))
    my_eng.execute(text(f"DELETE FROM {'testResults'}"))
    my_eng.execute(text(f"DROP TABLE IF EXISTS {'MasterDataset'}"))
    my_eng.execute(text(f"DROP TABLE IF EXISTS {'DataSets'}"))


def get_pipeline_from_name(name):
    """
    Get the pipeline ID from its name in the database.

    Args:
        name (str): Name of the pipeline.

    Returns:
        int: ID of the pipeline.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    NodeId = int(
        pd.read_sql(
            text(f"SELECT id FROM FLpipeline WHERE name = '{name}'"), my_eng
        ).iloc[0, 0]
    )
    return NodeId


def get_pipeline_confusion_matrix(pipeline_id):
    """
    Get the global confusion matrix for a pipeline based on test results.

    Args:
        pipeline_id (int): ID of the pipeline.

    Returns:
        dict: A dictionary representing the global confusion matrix.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    data = pd.read_sql(
        text(
            f"SELECT confusionmatrix FROM testResults WHERE pipelineid = '{pipeline_id}'"), my_eng
    )

    # Convert the column of strings into a list of dictionaries representing confusion matrices
    confusion_matrices = [
        json.loads(matrix.replace("'", "\"")) for matrix in data['confusionmatrix']
    ]

    # Initialize variables for global confusion matrix
    global_TP = global_FP = global_FN = global_TN = 0

    # Iterate through each dictionary and sum the corresponding values for each category
    for matrix in confusion_matrices:
        global_TP += matrix['TP']
        global_FP += matrix['FP']
        global_FN += matrix['FN']
        global_TN += matrix['TN']

    # Create a global confusion matrix as a dictionary
    global_confusion_matrix = {
        'TP': global_TP,
        'FP': global_FP,
        'FN': global_FN,
        'TN': global_TN
    }
    # Return the list of dictionaries representing confusion matrices
    return global_confusion_matrix


def get_node_confusion_matrix(pipeline_id, node_name):
    """
    Get the confusion matrix for a specific node in a pipeline based on test results.

    Args:
        pipeline_id (int): ID of the pipeline.
        node_name (str): Name of the node.

    Returns:
        dict: A dictionary representing the confusion matrix for the specified node.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    data = pd.read_sql(
        text(
            f"SELECT confusionmatrix FROM testResults WHERE pipelineid = '{pipeline_id}' AND nodename = '{node_name}'"), my_eng
    )

    # Convert the column of strings into a list of dictionaries representing confusion matrices
    confusion_matrices = [
        json.loads(matrix.replace("'", "\"")) for matrix in data['confusionmatrix']
    ]

    # Return the list of dictionaries representing confusion matrices
    return confusion_matrices[0]


def get_pipeline_result(pipeline_id):
    """
    Get the test results for a pipeline.

    Args:
        pipeline_id (int): ID of the pipeline.

    Returns:
        pandas.DataFrame: DataFrame containing test results for the specified pipeline.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    data = pd.read_sql(
        text(
            f"SELECT * FROM testResults WHERE pipelineid = '{pipeline_id}'"), my_eng
    )
    return data
