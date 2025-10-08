import mysql.connector
from sqlalchemy import create_engine, text
from configparser import ConfigParser
import yaml
import pkg_resources
import os

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load configuration from the config file
config_file_path = os.path.join(current_directory, 'db_config.ini')

config = ConfigParser()
config.read(config_file_path)
mysql_config = config['mysql']



connection_string = (
    f"mysql+mysqlconnector://{mysql_config['user']}:{mysql_config['password']}@"
    f"{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
)

eng = create_engine(
    connection_string,
    execution_options={"autocommit": True},
)

my_eng = eng.connect()