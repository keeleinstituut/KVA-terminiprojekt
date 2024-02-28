import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import os


load_dotenv()


def get_iate_tokens():
    """Request and return IATE tokens using environment variables for credentials."""

    IATE_USERNAME = os.getenv('IATE_USERNAME')
    IATE_PASSWORD = os.getenv('IATE_PASSWORD')

    base_url = "https://iate.europa.eu/uac-api/oauth2/token"

    params = {
        'username': IATE_USERNAME,
        'password': IATE_PASSWORD,
        'grant_type': 'password'
    }

    full_url = f"{base_url}?{urlencode(params)}"

    headers = {
        'Accept': 'application/vnd.iate.token+json; version=2',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(full_url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to retrieve tokens', 'details': response.text}


def refresh_iate_tokens(refresh_token):
    """Refresh IATE tokens using the provided refresh token."""

    base_url = "https://iate.europa.eu/uac-api/oauth2/token"

    params = {
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }

    full_url = f"{base_url}?{urlencode(params)}"

    headers = {
        'Accept': 'application/vnd.iate.token+json; version=2',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(full_url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to refresh tokens', 'details': response.text}