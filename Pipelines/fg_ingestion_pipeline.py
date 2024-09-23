import pandas as pd
from components.fg_ingestion import fgIngestion
from constants import *
import hopsworks
import configparser

STAGE_NAME = "Feature Group Ingestion"

class fgIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            fgIngestion_obj = fgIngestion()
            
            fgIngestion_obj.add_id_and_event_time_columns()
    
            fgIngestion_obj.fill_missing_string_values()
            
            fgIngestion_obj.convert_string_to_datetime()
            
            # fgIngestion_obj.save_dataframes_to_db()
            
            project = fgIngestion_obj.hopswork_login()
            fs = project.get_feature_store()
            fgIngestion_obj.process_feature_groups(fs)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            raise e

    
if __name__ == '__main__':
    try:
        print(">>>>>> Stage started <<<<<< :",STAGE_NAME)
        obj = fgIngestionPipeline()
        obj.main()
        print(">>>>>> Stage completed <<<<<<", STAGE_NAME)
    except Exception as e:
        print(e)
        raise e