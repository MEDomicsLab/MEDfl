import sys
import sqlite3
import pandas as pd
from configparser import ConfigParser
import os
import ast 

from MEDfl.LearningManager.utils import *


def main(csv_file_path):
    try:
        # Get the directory of the current script
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Load configuration from the config file
        # config_file_path = os.path.join(current_directory, 'sqllite_config.ini')*

        config_file_path = load_db_config()

        # config = ConfigParser()
        # config.read(config_file_path)
        # sqlite_config = config['sqllite']

        sqlite_config = config_file_path 


        print('Im here !')

        # Connect to SQLite database (it will be created if it doesn't exist)
        database_path = sqlite_config['database']
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Drop each table if it exists
        tables = ['Networks', 'FLsetup', 'Nodes', 'DataSets', 'FLpipeline', 'testResults', 'FedDatasets']
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        # Create Networks table
        cursor.execute(
            "CREATE TABLE Networks( \
                         NetId INTEGER PRIMARY KEY AUTOINCREMENT, \
                         NetName TEXT \
                         );"
        )

        # Create FLsetup table
        cursor.execute("CREATE TABLE FLsetup (\
        FLsetupId INTEGER PRIMARY KEY AUTOINCREMENT,\
        name TEXT NOT NULL, \
        description TEXT NOT NULL,\
        creation_date TEXT NOT NULL,\
        NetId INTEGER NOT NULL,\
        column_name TEXT\
        )")

        # Create Nodes table
        cursor.execute("CREATE TABLE Nodes ( \
         NodeId INTEGER PRIMARY KEY AUTOINCREMENT,\
         NodeName TEXT,\
         train BOOLEAN DEFAULT 1,\
         NetId INTEGER\
        )")

        data_df = pd.read_csv(csv_file_path)
        columns = data_df.columns.tolist()
        column_map = {"object": "TEXT", "int64": "INTEGER", "float64": "REAL"}
        sub_query = ", ".join(f"{col} {column_map[str(data_df[col].dtype)]}" for col in columns)

        # Create Datasets table by getting columns from the master csv file
        cursor.execute(
            f"CREATE TABLE DataSets( \
                         DataSetId INTEGER PRIMARY KEY AUTOINCREMENT, \
                         DataSetName TEXT, \
                         NodeId INTEGER,\
                         {sub_query}\
                         )"
        )

        # Create FLpipeline table
        cursor.execute("CREATE TABLE FLpipeline(\
         id INTEGER PRIMARY KEY AUTOINCREMENT,\
         name TEXT NOT NULL, \
         description TEXT NOT NULL,\
         creation_date TEXT NOT NULL,\
         results TEXT NOT NULL\
        ) ")

        # Create test results table
        cursor.execute("CREATE TABLE testResults(\
         pipelineId INTEGER,\
         nodename TEXT NOT NULL, \
         confusionmatrix TEXT,\
         accuracy REAL,\
         sensivity REAL,\
         ppv REAL,\
         npv REAL,\
         f1score REAL,\
         fpr REAL,\
         tpr REAL, \
         PRIMARY KEY (pipelineId, nodename)\
        ) ")

        # Create FederatedDataset table
        cursor.execute("CREATE TABLE FedDatasets (\
         FedId INTEGER PRIMARY KEY AUTOINCREMENT,\
         FLsetupId INTEGER,\
         FLpipeId INTEGER,\
         name TEXT NOT NULL\
        )")

        # Commit and close the cursor
        conn.commit()
        cursor.close()
        conn.close()

    except sqlite3.Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    csv_file_path = sys.argv[1]
    main(csv_file_path)
