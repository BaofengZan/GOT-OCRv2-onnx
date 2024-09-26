import os
import requests


OWNER = 'wangzhaode'
REPO = 'qwen-1.8b-onnx'

api_url = f'https://api.github.com/repos/{OWNER}/llm-export/releases/tags/{REPO}'

# Get the latest release information
response = requests.get(api_url)
if response.status_code != 200:
    raise Exception(f'Error fetching release information: {response.status_code}')

release_data = response.json()

# Create a directory to save the assets
os.makedirs(REPO, exist_ok=True)

# Download each asset in the release
for asset in release_data['assets']:
    asset_url = asset['browser_download_url']
    asset_name = asset['name']

    print(f'Downloading {asset_name}...')

    asset_response = requests.get(asset_url, stream=True)
    if asset_response.status_code == 200:
        asset_path = os.path.join(REPO, asset_name)
        with open(asset_path, 'wb') as asset_file:
            for chunk in asset_response.iter_content(chunk_size=1024):
                if chunk:
                    asset_file.write(chunk)
        print(f'{asset_name} downloaded successfully.')
    else:
        print(f'Failed to download {asset_name}. Status code: {asset_response.status_code}')
