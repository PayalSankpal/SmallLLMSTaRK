import os
from dotenv import load_dotenv
import requests

load_dotenv()
keys = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
if not keys:
    print("No keys found!")
else:
    key = keys[0]
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers)
    try:
        data = r.json()
        models = [m['id'] for m in data.get('data', []) if 'llama-3' in m['id'].lower()]
        print("Available Llama 3 models:")
        for m in models: print(m)
    except Exception as e:
        print(r.text)
