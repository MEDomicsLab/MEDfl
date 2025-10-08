# src/MEDfl/NetManager/network.py

from MEDfl.LearningManager.utils import *
from .net_helper import *
from .net_manager_queries import (CREATE_MASTER_DATASET_TABLE_QUERY,
                                  CREATE_DATASETS_TABLE_QUERY,
                                  DELETE_NETWORK_QUERY,
                                  INSERT_NETWORK_QUERY, LIST_ALL_NODES_QUERY,
                                  UPDATE_NETWORK_QUERY, GET_NETWORK_QUERY)
from .node import Node
import pandas as pd
from MEDfl.LearningManager.utils import params

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

class Network:
    """
    A class representing a network.

    Attributes:
        name (str): The name of the network.
        mtable_exists (int): An integer flag indicating whether the MasterDataset table exists (1) or not (0).
    """

    def __init__(self, name: str = ""):
        """
        Initialize a Network instance.

        Parameters:
            name (str): The name of the network.
        """
        self.name = name
        self.mtable_exists = int(master_table_exists())
        self.validate()

        db_manager = DatabaseManager() 
        db_manager.connect() 
        self.eng = db_manager.get_connection()

    def validate(self):
        """Validate name"""

        if not isinstance(self.name, str):
            raise TypeError("name argument must be a string")

    def create_network(self):
        """Create a new network in the database."""
        try:
            print(self.name)
            self.eng.execute(text(INSERT_NETWORK_QUERY), {"name": self.name})
            self.id = self.get_netid_from_name(self.name)
        except SQLAlchemyError as e:
            print(f"Error creating network: {e}")

    def use_network(self, network_name: str):
        """Use a network in the database.

        Parameters:
            network_name (str): The name of the network to use.

        Returns:
            Network or None: An instance of the Network class if the network exists, else None.
        """
        try:
            network = pd.read_sql(
                text(GET_NETWORK_QUERY),
                self.eng,
                params={"name": network_name}
            )
            if not network.empty:
                self.name = network.iloc[0]['NetName']
                self.id = network.iloc[0]['NetId']
                self.mtable_exists = int(master_table_exists())
                self.validate()
                return self
            else:
                return None
        except SQLAlchemyError as e:
            print(f"Error using network: {e}")
            return None

    def delete_network(self):
        """Delete the network from the database."""
        try:
            self.eng.execute(text(DELETE_NETWORK_QUERY), {"name": self.name})
        except SQLAlchemyError as e:
            print(f"Error deleting network: {e}")

    def update_network(self, FLsetupId: int):
        """Update the network's FLsetupId in the database.

        Parameters:
            FLsetupId (int): The FLsetupId to update.
        """
        try:
            self.eng.execute(
                text(UPDATE_NETWORK_QUERY),
                {"FLsetupId": FLsetupId, "id": self.id}
            )
        except SQLAlchemyError as e:
            print(f"Error updating network: {e}")

    def add_node(self, node: Node):
        """Add a node to the network.

        Parameters:
            node (Node): The node to add.
        """
        node.create_node(self.id)

    def list_allnodes(self):
        """List all nodes in the network.

        Returns:
            DataFrame: A DataFrame containing information about all nodes in the network.
        """
        try:
            query = text(LIST_ALL_NODES_QUERY)
            result_proxy = self.eng.execute(query, name=self.name)
            result_df = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
            return result_df
        except SQLAlchemyError as e:
            print(f"Error listing all nodes: {e}")
            return pd.DataFrame()

    def create_master_dataset(self, path_to_csv: str = params['path_to_master_csv']):
        """
        Create the MasterDataset table and insert dataset values.

        :param path_to_csv: Path to the CSV file containing the dataset.
        """
        try:
            print(path_to_csv)
            data_df = pd.read_csv(path_to_csv)

            if self.mtable_exists != 1:
                columns = data_df.columns.tolist()
                columns_str = ",\n".join(
                    [
                        f"{col} {column_map[str(data_df[col].dtype)]}"
                        for col in columns
                    ]
                )
                self.eng.execute(
                    text(CREATE_MASTER_DATASET_TABLE_QUERY.format(columns_str))
                )
                self.eng.execute(text(CREATE_DATASETS_TABLE_QUERY.format(columns_str)))

                # Process data
                data_df = process_eicu(data_df)

                # Insert data in batches
                batch_size = 1000  # Adjust as needed
                for start_idx in range(0, len(data_df), batch_size):
                    batch_data = data_df.iloc[start_idx:start_idx + batch_size]
                    insert_query = f"INSERT INTO MasterDataset ({', '.join(columns)}) VALUES ({', '.join([':' + col for col in columns])})"
                    data_to_insert = batch_data.to_dict(orient='records')
                    self.eng.execute(text(insert_query), data_to_insert)

                self.mtable_exists = 1
        except SQLAlchemyError as e:
            print(f"Error creating master dataset: {e}")
            
    @staticmethod
    def list_allnetworks():
        """List all networks in the database.

        Returns:
            DataFrame: A DataFrame containing information about all networks in the database.
        """
        try:
            db_manager = DatabaseManager() 
            db_manager.connect() 
            my_eng = db_manager.get_connection() 

            result_proxy = my_eng.execute("SELECT * FROM Networks")
            result = result_proxy.fetchall()
            return pd.DataFrame(result, columns=result_proxy.keys())
        except SQLAlchemyError as e:
            print(f"Error listing all networks: {e}")
            return pd.DataFrame()
    
    def get_netid_from_name(self, name):
        """Get network ID from network name."""
        try:
            result = self.eng.execute(text("SELECT NetId FROM Networks WHERE NetName = :name"), {"name": name}).fetchone()
            if result:
                return result[0]
            else:
                return None
        except SQLAlchemyError as e:
            print(f"Error fetching network ID: {e}")
            return None
