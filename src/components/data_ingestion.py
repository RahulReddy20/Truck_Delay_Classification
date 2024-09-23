import configparser
import time
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError
from constants import *
from utils.config_utils import *
from utils.database_utils import *


class DataIngestion:
    def __init__(self):
        self.config = read_config(CONFIG_FILE_PATH)
        self.db_config = get_db_config(self.config)
        self.engine = None

    def create_database_if_not_exists(self):
        """Check if the target database exists, and create it if not."""
        default_engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/postgres",
            isolation_level='AUTOCOMMIT'
        )
        
        with default_engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": self.db_config['dbname']}
            ).fetchone()
            
            if not result:
                conn.execute(text(f"CREATE DATABASE {self.db_config['dbname']}"))
                print(f"Database {self.db_config['dbname']} created.")
            else:
                print(f"Database {self.db_config['dbname']} already exists.")

        default_engine.dispose()

    def wait_for_database_ready(self):
        """Wait for the database to be ready by continuously attempting to connect."""
        while True:
            try:
                self.engine = connect(self.db_config)
                with self.engine.connect():
                    print(f"Database {self.db_config['dbname']} is ready.")
                    return self.engine
            except OperationalError:
                print(f"Waiting for the database {self.db_config['dbname']} to be ready...")
                time.sleep(2)

    def get_csv_urls(self):
        """Retrieve a list of CSV URLs from a GitHub repository page."""
        response = requests.get(self.config['DATA']['github_url'])
        soup = BeautifulSoup(response.text, 'html.parser')

        csv_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.endswith('.csv'):
                csv_links.append(f"https://raw.githubusercontent.com{href.replace('/blob/', '/')}")

        return list(set(csv_links))

    def process_csv_file(self, csv_url):
        """Process a CSV file from the given URL and insert its data into a database table."""
        if self.engine is None:
            connect(self.db_config)

        csv_filename = os.path.basename(csv_url).replace('.csv', '')
        table_name = csv_filename  

        df = pd.read_csv(csv_url)
        inspector = inspect(self.engine)

        if not inspector.has_table(table_name):
            df.to_sql(table_name, self.engine, index=False)
            print(f"Table {table_name} created and data inserted.")
        else:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            print(f"Data replaced in the existing table {table_name}.")

    def process_all_csv_files(self):
        """Retrieve CSV URLs and process each CSV file into the database."""
        csv_urls = self.get_csv_urls()
        for csv_url in csv_urls:
            self.process_csv_file(csv_url)
