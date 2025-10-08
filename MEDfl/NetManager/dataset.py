import pandas as pd
from sqlalchemy import text

from .net_helper import *
from .net_manager_queries import (DELETE_DATASET, INSERT_DATASET,
                                  SELECT_ALL_DATASET_NAMES)
from MEDfl.NetManager.database_connector import DatabaseManager

class DataSet:
    def __init__(self, name: str, path: str, engine=None):
        """
        Initialize a DataSet object.

        :param name: The name of the dataset.
        :type name: str
        :param path: The file path of the dataset CSV file.
        :type path: str
        """
        self.name = name
        self.path = path
        db_manager = DatabaseManager()
        db_manager.connect()
        self.engine = db_manager.get_connection()

    def validate(self):
        """
        Validate name and path attributes.

        :raises TypeError: If name or path is not a string.
        """
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.path, str):
            raise TypeError("path argument must be a string")

    def upload_dataset(self, NodeId=-1):
        """
        Upload the dataset to the database.

        :param NodeId: The NodeId associated with the dataset.
        :type NodeId: int

        Notes:
        - Assumes the file at self.path is a valid CSV file.
        - The dataset is uploaded to the 'DataSets' table in the database.
        """

        data_df = pd.read_csv(self.path)
        nodeId = NodeId
        columns = data_df.columns.tolist()
        

        data_df = process_eicu(data_df)
        for index, row in data_df.iterrows():
            query_1 = "INSERT INTO DataSets(DataSetName,nodeId," + "".join(
                f"{x}," for x in columns
            )
            query_2 = f" VALUES ('{self.name}',{nodeId}, " + "".join(
                f"{is_str(data_df, row, x)}," for x in columns
            )
            query = query_1[:-1] + ")" + query_2[:-1] + ")"
             
            self.engine.execute(text(query))

    def delete_dataset(self):
        """
        Delete the dataset from the database.

        Notes:
        - Assumes the dataset name is unique in the 'DataSets' table.
        """
        self.engine.execute(text(DELETE_DATASET), {"name": self.name})

    def update_data(self):
        """
        Update the data in the dataset.

        Not implemented yet.
        """
        pass

    @staticmethod
    def list_alldatasets(engine):
        """
        List all dataset names from the 'DataSets' table.

        :returns: A DataFrame containing the names of all datasets in the 'DataSets' table.
        :rtype: pd.DataFrame
        """
        res = pd.read_sql(text(SELECT_ALL_DATASET_NAMES), engine)
        return res
