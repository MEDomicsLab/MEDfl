import pandas as pd

from .net_helper import *
from .net_manager_queries import *
from MEDfl.LearningManager.utils import params
from MEDfl.NetManager.database_connector import DatabaseManager

from sqlalchemy import text, exc


class Node:
    """
    A class representing a node in the network.

    Attributes:
        name (str): The name of the node.
        train (int): An integer flag representing whether the node is used for training (1) or testing (0).
        test_fraction (float, optional): The fraction of data used for testing when train=1. Default is 0.2.
    """

    def __init__(
        self, name: str, train: int, test_fraction: float = 0.2, engine=None
    ):
        """
        Initialize a Node instance.

        Parameters:
            name (str): The name of the node.
            train (int): An integer flag representing whether the node is used for training (1) or testing (0).
            test_fraction (float, optional): The fraction of data used for testing when train=1. Default is 0.2.
        """
        self.name = name
        self.train = train
        self.test_fraction = 1.0 if self.train == 0 else test_fraction
       

        db_manager = DatabaseManager() ; 
        db_manager.connect() ; 
        self.engine = db_manager.get_connection()

    def validate(self):
        """Validate name, train, test_fraction"""
        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

        if not isinstance(self.train, int):
            raise TypeError("train argument must be an int")

        if not isinstance(self.test_fraction, float):
            raise TypeError("test_fraction argument must be a float")

    def create_node(self, NetId: int):
        """Create a node in the database.
        Parameters:
            NetId (int): The ID of the network to which the node belongs.

        Returns:
            None
        """
        self.engine.execute(
            text(INSERT_NODE_QUERY.format(self.name, NetId, self.train))
        )

    def delete_node(self):
        """Delete the node from the database."""
        self.engine.execute(text(DELETE_NODE_QUERY.format(self.name)))

    def check_dataset_compatibility(self, data_df):
        """Check if the dataset is compatible with the master dataset.
        Parameters:
            data_df (DataFrame): The dataset to check.

        Returns:
            None
        """
        if master_table_exists() != 1:
            print("MasterDataset doesn't exist")
        else:
            columns = data_df.columns.tolist()
            
            # get master_dataset columns
            result_proxy = self.engine.execute(SELECT_MASTER_COLUMNS_QUERY)
            master_table_columns = result_proxy.keys()


            assert [x == y for x, y in zip(master_table_columns, columns)]

    def update_node(self):
        """Update the node information (not implemented)."""
        pass

    def get_dataset(self, column_name: str = None):
        """Get the dataset for the node based on the given column name.
        Parameters:
            column_name (str, optional): The column name to filter the dataset. Default is None.

        Returns:
            DataFrame: The dataset associated with the node.
        """
        NodeId = get_nodeid_from_name(self.name)
        if column_name is not None:
            query = text(SELECT_DATASET_BY_COLUMN_QUERY.format(column_name, self.name))
        else:
            query = text(SELECT_DATASET_BY_NODE_ID_QUERY.format(NodeId))

        result_proxy = self.engine.execute(query)
        node_dataset = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())

        return node_dataset

    def upload_dataset(self, dataset_name: str, path_to_csv: str = params['path_to_test_csv']):
        """Upload the dataset to the database for the node.

        Parameters:
            dataset_name (str): The name of the dataset.
            path_to_csv (str, optional): Path to the CSV file containing the dataset. Default is the path in params.

        Returns:
            None
        """
        try:
            data_df = pd.read_csv(path_to_csv)
            nodeId = get_nodeid_from_name(self.name)
            columns = data_df.columns.tolist()
            self.check_dataset_compatibility(data_df)

            data_df = process_eicu(data_df)

            # Insert data in batches
            batch_size = 1000  # Adjust as needed
            for start_idx in range(0, len(data_df), batch_size):
                batch_data = data_df.iloc[start_idx:start_idx + batch_size]
                insert_query = f"INSERT INTO Datasets (DataSetName, NodeId, {', '.join(columns)}) VALUES (:dataset_name, :nodeId, {', '.join([':' + col for col in columns])})"
                data_to_insert = batch_data.to_dict(orient='records')
                params = [{"dataset_name": dataset_name, "nodeId": nodeId, **row} for row in data_to_insert]
                self.engine.execute(text(insert_query), params)
        except exc.SQLAlchemyError as e:
            print(f"Error uploading dataset: {e}")

    def assign_dataset(self, dataset_name:str):
        """Assigning existing dataSet to node
        Parameters:
            dataset_name (str): The name of the dataset to assign.

        Returns:
            None
        """

        nodeId = get_nodeid_from_name(self.name)
        query = f"UPDATE DataSets SET nodeId = {nodeId} WHERE DataSetName = '{dataset_name}'"
        self.engine.execute(text(query))

    def unassign_dataset(self, dataset_name:str):
        """unssigning existing dataSet to node
        Parameters:
            dataset_name (str): The name of the dataset to assign.

        Returns:
            None
        """

        query = f"UPDATE DataSets SET nodeId = {-1} WHERE DataSetName = '{dataset_name}'"
        self.engine.execute(text(query))

    def list_alldatasets(self):
        """List all datasets associated with the node.
        Returns:
            DataFrame: A DataFrame containing information about all datasets associated with the node.
        
        """
        return pd.read_sql(
            text(SELECT_ALL_DATASETS_QUERY.format(self.name)), my_eng
        )

    @staticmethod
    def list_allnodes():
        """List all nodes in the database.
        Returns:
            DataFrame: A DataFrame containing information about all nodes in the database.
        
        """
        query = text(SELECT_ALL_NODES_QUERY)
        res = pd.read_sql(query, my_eng)
        return res
