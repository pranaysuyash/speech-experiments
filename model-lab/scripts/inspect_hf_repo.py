from huggingface_hub import list_repo_files
import json
from huggingface_hub import hf_hub_download
import os

repo_id = "zai-org/GLM-TTS"
token = os.environ.get("HF_TOKEN")

print(f"Inspecting {repo_id} with token {token[:5]}...")

try:
    files = list_repo_files(repo_id, token=token)
    print("Files:", files)
    
    if "config.json" in files:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
        with open(config_path, "r") as f:
            config = json.load(f)
        print("Config:", json.dumps(config, indent=2))
    else:
        print("No config.json found!")
        
except Exception as e:
    print(f"Error: {e}")
