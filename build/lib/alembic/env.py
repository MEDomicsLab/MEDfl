from logging.config import fileConfig
import logging
from sqlalchemy import engine_from_config, create_engine
from sqlalchemy import pool
import sys
import os
from alembic import context

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.base import my_eng

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.
def configure_logger(name):
    # This is the standard logging configuration
    logging.config.fileConfig(
        'alembic_logging.ini',  # Path to your logging configuration file
        defaults={
            'logfilename': 'alembic.log',  # Log file name
        },
        disable_existing_loggers=False,
    )

    return logging.getLogger(name)



def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    pass

def run_migrations_online():
    """Run migrations in 'online' mode."""
    pass

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
