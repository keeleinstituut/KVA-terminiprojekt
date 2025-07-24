import json
import os

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
        print('load_config')
    return config

config = load_config(os.getenv('APP_CONFIG'))

