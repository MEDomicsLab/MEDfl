from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sqlalchemy import text

import torch
import pandas as pd
from torch.utils.data import TensorDataset
import numpy as np

from MEDfl.NetManager.database_connector import DatabaseManager


def is_str(data_df, row, x):
    """
    Check if a column in a DataFrame is of type 'object' and convert the value accordingly.

    Args:
        data_df (pandas.DataFrame): DataFrame containing the data.
        row (pandas.Series): Data row.
        x (str): Column name.

    Returns:
        str or float: Processed value based on the column type.
    """
    if data_df[x].dtype == "object":
        x = f"'{row[x]}'"
    else:
        x = row[x]
    return x


def process_eicu(data_df):
    """
    Process eICU data by filling missing values with mean and replacing NaNs with 'Unknown'.

    Args:
        data_df (pandas.DataFrame): Input data.

    Returns:
        pandas.DataFrame: Processed data.
    """
    # Identify numeric and non-numeric columns
    numeric_columns = data_df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = data_df.select_dtypes(exclude=[np.number]).columns

    # Fill NaN in numeric columns with mean
    data_df[numeric_columns] = data_df[numeric_columns].fillna(
        data_df[numeric_columns].mean())

    # Fill NaN in non-numeric columns with 'Unknown'
    data_df[non_numeric_columns] = data_df[non_numeric_columns].fillna(
        'Unknown')

    try:
        data_df = data_df.reset_index(drop=True)
    except:
        pass

    return data_df


# remove indiserd columns after reading from the DB
def process_data_after_reading(data, output, fill_strategy="mean",  fit_encode=[], to_drop=[]):
    """
    Process data after reading from the database, including encoding, dropping columns, and creating a PyTorch TensorDataset.

    Args:
        data (pandas.DataFrame): Input data.
        output (str): Output column name.
        fill_strategy (str, optional): Imputation strategy for missing values. Default is "mean".
        fit_encode (list, optional): List of columns to be label-encoded. Default is an empty list.
        to_drop (list, optional): List of columns to be dropped from the DataFrame. Default is an empty list.

    Returns:
        torch.utils.data.TensorDataset: Processed data as a PyTorch TensorDataset.
    """

    # Check if there is a DataSet assigned to the node
    if (len(data) == 0):
        raise "Node doesn't Have dataSet"

    encoder = LabelEncoder()
    # En Code some columns
    for s in fit_encode:
        try:
            data[s] = encoder.fit_transform(data[s])
        except:
            raise print(s)

    # The output of the DATA
    y = data[output]

    X = data

    # remove indisered columns when reading the dataframe from the DB
    for column in to_drop:
        try:
            X = X.drop(
                [column], axis=1
            )
        except Exception as e:
            raise e

    # Get the DATAset Features
    features = [col for col in X.columns if col != output]

    # Impute missing values using the mean strategy
    try:
        imputer = SimpleImputer(strategy=fill_strategy)
        X[features] = imputer.fit_transform(X[features])
    except:
        print()

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    data = TensorDataset(X, y)

    return data


def get_nodeid_from_name(name):
    """
    Get the NodeId from the Nodes table based on the NodeName.

    Args:
        name (str): Node name.

    Returns:
        int or None: NodeId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    result_proxy = my_eng.execute(f"SELECT NodeId FROM Nodes WHERE NodeName = '{name}'")
    NodeId = int(result_proxy.fetchone()[0])
    return NodeId


def get_netid_from_name(name):
    """
    Get the Network Id from the Networks table based on the NetName.

    Args:
        name (str): Network name.

    Returns:
        int or None: NetId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        result_proxy = my_eng.execute(f"SELECT NetId FROM Networks WHERE NetName = '{name}'")
        NetId = int(result_proxy.fetchone()[0])
    except:
        NetId = None
    return NetId


def get_flsetupid_from_name(name):
    """
    Get the FLsetupId from the FLsetup table based on the FL setup name.

    Args:
        name (str): FL setup name.

    Returns:
        int or None: FLsetupId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        
         result_proxy = my_eng.execute(f"SELECT FLsetupId FROM FLsetup WHERE name = '{name}'")
         id = int(result_proxy.fetchone()[0])
   
    except:
        id = None
    return id


def get_flpipeline_from_name(name):
    """
    Get the FLpipeline Id from the FLpipeline table based on the FL pipeline name.

    Args:
        name (str): FL pipeline name.

    Returns:
        int or None: FLpipelineId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:

        result_proxy = my_eng.execute(f"SELECT id FROM FLpipeline WHERE name = '{name}'")
        id = int(result_proxy.fetchone()[0])
    except:
        id = None
    return id


def get_feddataset_id_from_name(name):
    """
    Get the Federated dataset Id from the FedDatasets table based on the federated dataset name.

    Args:
        name (str): Federated dataset name.

    Returns:
        int or None: FedId or None if not found.
    """
    db_manager = DatabaseManager()
    db_manager.connect()
    my_eng = db_manager.get_connection()

    try:
        
        result_proxy = my_eng.execute(f"SELECT FedId FROM FedDatasets WHERE name = '{name}'")
        id = int(result_proxy.fetchone()[0])
    except:
        id = None
    return id


def master_table_exists():
    """
    Check if the MasterDataset table exists in the database.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    try:
        db_manager = DatabaseManager()
        db_manager.connect()
        my_eng = db_manager.get_connection()

        # SQLite-specific query to check if table exists
        sql_query = text("SELECT name FROM sqlite_master WHERE type='table' AND name='MasterDataset'")
        result = my_eng.execute(sql_query)
        exists = result.fetchone() is not None
        return exists

    except Exception as e:
        print(f"Error checking MasterDataset table existence: {e}")
        return False

