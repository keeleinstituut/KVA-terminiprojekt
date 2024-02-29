import requests
from dotenv import load_dotenv
import os
import json


load_dotenv()

COLLINS_API_KEY = os.getenv('COLLINS_PASSWORD')

url = 'https://api.collinsdictionary.com/api/v1/dictionaries'

headers = {
    'Accept': 'application/json',
    'accessKey': COLLINS_API_KEY,
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Available dictionaries:")
    print(json.dumps(data, indent=4))
else:
    print(f"Failed to retrieve dictionaries, status code: {response.status_code}")