import pandas as pd
from sqlalchemy import create_engine, inspect
import psycopg2
import requests
from bs4 import BeautifulSoup
import os
import time
from constants import *
import joblib

def create_database_if_not_exists(config):
    conn = psycopg2.connect(
        dbname='postgres', user=config['user'], password=config['password'], host=config['host'], port=config['port']
    )
    conn.autocommit = True
    cursor = conn.cursor()

    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{config['dbname']}'")
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(f"CREATE DATABASE {config['dbname']}")
        print(f"Database {config['dbname']} created.")
    else:
        print(f"Database {config['dbname']} already exists.")

    cursor.close()
    conn.close()

def wait_for_database_ready(config):
    while True:
        try:
            conn = psycopg2.connect(
                dbname=config['dbname'], user=config['user'], password=config['password'], host=config['host'], port=config['port']
            )
            conn.close()
            break
        except psycopg2.OperationalError:
            print(f"Waiting for the database {config['dbname']} to be ready...")
            time.sleep(2)  

def get_csv_urls(github_folder_url):
    response = requests.get(github_folder_url)
    
    soup = BeautifulSoup(response.text, 'html.parser')

    csv_links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.endswith('.csv'):
            csv_links.append(f"https://raw.githubusercontent.com{href.replace('/blob/', '/')}")
    csv_links = list(set(csv_links))
    
    return csv_links

def process_csv_file(csv_url, engine):
    csv_filename = os.path.basename(csv_url).replace('.csv', '')
    table_name = csv_filename  

    df = pd.read_csv(csv_url)

    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        df.to_sql(table_name, engine, index=False)
        print(f"Table {table_name} created and data inserted.")
    else:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data appended to the existing table {table_name}.")

def save_tables_to_joblib(engine, output_file):
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    table_dataframes = {}

    for table_name in tables:
        # Load table data into a DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        table_dataframes[table_name] = df

    # Save the dictionary of DataFrames to a .joblib file
    joblib.dump(table_dataframes, output_file)
    print(f"All tables saved to {output_file} as a dictionary of DataFrames.")

# Main script execution
if __name__ == "__main__":
    
    create_database_if_not_exists(DB_CONFIG)

    wait_for_database_ready(DB_CONFIG)

    engine = create_engine(f'postgresql+psycopg2://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}:{DB_CONFIG["port"]}/{DB_CONFIG["dbname"]}')

    csv_urls = get_csv_urls(GITHUB_FOLDER_URL)

    for csv_url in csv_urls:
        print(f"Processing {csv_url}")
        process_csv_file(csv_url, engine)

    joblib_file_path = './Data/DataFrame/dataframes_dict.joblib'
    save_tables_to_joblib(engine, joblib_file_path)