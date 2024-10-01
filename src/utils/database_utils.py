from sqlalchemy import create_engine, text, inspect
import pandas as pd

def connect(db_config):
    """Establish a connection to the PostgreSQL database using the configuration."""
    try:
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        print("Connection to PostgreSQL successful")
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
    return engine
    
def read_table(engine, table_name):
    """Read a table from the connected database."""
    if engine is None:
        raise Exception("Database not connected.")
    
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql_query(text(query), engine)

def read_all_tables(engine):
    """Read all tables in the public schema of the connected database."""
    if engine is None:
        raise Exception("Database not connected.")
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    dataframes_dict = {table: pd.read_sql_table(table, engine) for table in tables}
    return dataframes_dict

def close_connection(engine):
    """Close the current database connection."""
    if engine:
        engine.dispose()
        print("Connection closed.")
        
def get_table_names(engine):
    """Inspect and list table names in the connected database."""
    if not engine:
        raise Exception("Database not connected.")
    
    inspector = inspect(engine)
    return inspector.get_table_names()