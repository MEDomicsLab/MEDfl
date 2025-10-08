from .net_helper import is_str

INSERT_DATASET = """
    INSERT INTO DataSets(DataSetName, NodeId, {columns})
    VALUES (:name, :NodeId, {values})
"""
DELETE_DATASET = """
    DELETE FROM DataSets WHERE DataSetName = :name
"""

SELECT_ALL_DATASET_NAMES = """
    SELECT DISTINCT DataSetName,NodeId FROM DataSets
"""

SELECT_DATASET_BY_NAME = """
    SELECT * FROM DataSets WHERE DataSetName = :name
"""

# node queries
# sql_queries.py

INSERT_NODE_QUERY = (
    "INSERT INTO Nodes(NodeName,NetId,train) VALUES ('{}',{}, {})"
)
DELETE_NODE_QUERY = "DELETE FROM Nodes WHERE NodeName = '{}'"
SELECT_MASTER_COLUMNS_QUERY = "SELECT * FROM MasterDataset LIMIT 1"
SELECT_DATASET_BY_COLUMN_QUERY = "SELECT * FROM MasterDataset WHERE {} = '{}'"
SELECT_DATASET_BY_NODE_ID_QUERY = "SELECT * FROM DataSets WHERE NodeId = {}"

SELECT_ALL_DATASETS_QUERY = "SELECT DISTINCT DataSetName,NodeName FROM DataSets,Nodes WHERE Nodes.NodeName = '{}' and Nodes.NodeId = DataSets.NodeId"
SELECT_ALL_NODES_QUERY = "SELECT * FROM Nodes"


# SQL query to insert a new network
INSERT_NETWORK_QUERY = "INSERT INTO Networks(NetName) VALUES (:name)"

# SQL query to delete a network
DELETE_NETWORK_QUERY = "DELETE FROM Networks WHERE NetName = '{name}'"

# SQL query to delete a network
GET_NETWORK_QUERY = "SELECT * FROM Networks WHERE NetName = '{name}'"


# SQL query to update a network
UPDATE_NETWORK_QUERY = (
    "UPDATE Networks SET FLsetupId = {FLsetupId} WHERE NetId = {id}"
)

# SQL query to retrieve all nodes for a network
LIST_ALL_NODES_QUERY = """
SELECT Nodes.NodeName, Networks.NetName 
FROM Nodes 
JOIN Networks ON Networks.NetId = Nodes.NetId 
WHERE Networks.NetName = :name
"""

# SQL query to create the MasterDataset table (SQLite-compatible)
CREATE_MASTER_DATASET_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS MasterDataset (
    PatientId INTEGER PRIMARY KEY AUTOINCREMENT,
    {}
);
"""


# SQL query to create the datasets table (SQLite-compatible)
CREATE_DATASETS_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS Datasets (
    DataSetId INTEGER PRIMARY KEY AUTOINCREMENT, 
    DataSetName VARCHAR(255), 
    NodeId INT,
    {}
);
"""



# SQL query to insert dataset values
INSERT_DATASET_VALUES_QUERY = "INSERT INTO MasterDataset({columns}, NodeId) VALUES ('{name}', {nodeId}, {values})"


# FL setup_queries
# sql_queries.py

CREATE_FLSETUP_QUERY = """
    INSERT INTO FLsetup (name, description, creation_date, NetId, column_name)
    VALUES (:name, :description, :creation_date, :net_id, :column_name)
"""

DELETE_FLSETUP_QUERY = """
    DELETE FROM FLsetup
    WHERE name = :name
"""

UPDATE_FLSETUP_QUERY = UPDATE_NETWORK_QUERY = (
    "UPDATE FLsetup SET column_name ='{column_name}' WHERE name ='{FLsetupName}'"
)


READ_SETUP_QUERY = """
    SELECT * FROM FLsetup
    WHERE FLsetupId = :flsetup_id
"""

READ_ALL_SETUPS_QUERY = """
    SELECT * FROM FLsetup
"""

READ_NETWORK_BY_ID_QUERY = """
    SELECT * FROM Networks
    WHERE NetId = :net_id
"""

READ_DISTINCT_NODES_QUERY = """
SELECT DISTINCT {} FROM MasterDataset 
"""


# FederatedDataset Queries
INSERT_FLDATASET_QUERY = (
    "INSERT INTO FedDatasets(name, FLsetupId) VALUES (:name, :FLsetupId)"
)
DELETE_FLDATASET_BY_SETUP_AND_PIPELINE_QUERY = "DELETE FROM FedDatasets WHERE FLsetupId = :FLsetupId AND FLpipeId = :FLpipeId"


UPDATE_FLDATASET_QUERY = (
    "UPDATE FedDatasets SET FLpipeId = :FLpipeId WHERE FedId = :FedId"
)
SELECT_FLDATASET_BY_NAME_QUERY = "SELECT * FROM FedDatasets WHERE name = :name"

CREATE_FLPIPELINE_QUERY = """
INSERT INTO FLpipeline (name, description, creation_date, results)
VALUES ('{name}', '{description}', '{creation_date}', '{result}')
"""
DELETE_FLPIPELINE_QUERY = "DELETE FROM FLpipeline WHERE name = '{name}'"

SELECT_FLPIPELINE_QUERY = "SELECT FROM FLpipeline WHERE name = '{name}'"

CREATE_TEST_RESULTS_QUERY = """
INSERT INTO testResults (pipelineid, nodename, confusionmatrix, accuracy , sensivity, ppv , npv , f1score , fpr , tpr )
VALUES ('{pipelineId}', '{nodeName}', '{confusion_matrix}', '{accuracy}' , '{sensivity}' , '{ppv}' , '{npv}' , '{f1score}' , '{fpr}' , '{tpr}')
"""
