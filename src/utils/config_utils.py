import configparser

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def get_db_config(config):
    return {
        'dbname': config['DATABASE']['dbname'],
        'user': config['DATABASE']['user'],
        'password': config['DATABASE']['password'],
        'host': config['DATABASE']['host'],
        'port': config['DATABASE']['port']
    }