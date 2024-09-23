import pandas as pd
from components.data_ingestion import DataIngestion
from constants import *
import joblib
import os
from utils.database_utils import close_connection
STAGE_NAME = "Data Ingestion"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            data_ingestion = DataIngestion()
            data_ingestion.create_database_if_not_exists()
            engine = data_ingestion.wait_for_database_ready()
            csv_urls = data_ingestion.get_csv_urls()
            
            for csv_url in csv_urls:
                print(f"Processing {csv_url}")
                data_ingestion.process_csv_file(csv_url)
            # data = ingestion.read_all_tables()
            # joblib.dump(data, './Data/DataFrame/dataframes_dict.joblib')
            
            close_connection(engine)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = DataIngestionPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e