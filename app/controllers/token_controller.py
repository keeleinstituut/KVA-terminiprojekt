import time
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import os
import jwt


load_dotenv()

class TokenController:
    def __init__(self):
        self.base_url = "https://iate.europa.eu/uac-api/oauth2/token"
        self.username = os.getenv('IATE_USERNAME')
        self.password = os.getenv('IATE_PASSWORD')
        self.access_token = None
        self.refresh_token = None

    def get_iate_tokens(self):
        params = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password'
        }

        full_url = f"{self.base_url}?{urlencode(params)}"

        headers = {
            'Accept': 'application/vnd.iate.token+json; version=2',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.post(full_url, headers=headers)

        if response.status_code == 200:

            tokens = response.json()['tokens'][0]
            self.access_token = tokens.get('access_token')
            self.refresh_token = response.json().get('refresh_token')

            decoded_token = jwt.decode(tokens.get('access_token'), options={"verify_signature": False})
            token_expiration_time = decoded_token['exp']
            return tokens
        else:
            return {'error': 'Failed to retrieve tokens', 'details': response.text}

    def refresh_iate_tokens(self):
        if not self.refresh_token:
            return {'error': 'No refresh token available'}

        params = {
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }

        full_url = f"{self.base_url}?{urlencode(params)}"

        headers = {
            'Accept': 'application/vnd.iate.token+json; version=2',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.post(full_url, headers=headers)

        if response.status_code == 200:
            tokens = response.json()['tokens'][0]
            self.access_token = tokens.get('access_token')
            self.refresh_token = response.json().get('refresh_token')
            return tokens
        else:
            print(f"Failed to refresh tokens: {response.status_code} {response.text}")
            return {'error': 'Failed to refresh tokens', 'details': response.text}

    def get_access_token(self):
        if not self.access_token:
            tokens = self.get_iate_tokens()
            if 'error' in tokens:
                raise Exception(tokens['details'])
        return self.access_token