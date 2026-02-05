import os
from huggingface_hub import HfApi, login
from pathlib import Path

# Local path to your trained model
MODEL_PATH = Path("models/production/model_v1.1.0")

def upload_model():
    print("="*60)
    print("Hugging Face Model Upload")
    print("="*60)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("This script will upload your model to Hugging Face Hub.")
    print("You need a Hugging Face account and a User Access Token (Write).")
    print("Get your token here: https://huggingface.co/settings/tokens")
    print("-" * 60)
    
    # 1. Login
    token = input("Enter your Hugging Face Write Token: ").strip()
    if not token:
        print("Token is required.")
        return
        
    try:
        login(token=token, add_to_git_credential=True)
        print("Login successful!")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 2. Repo Details
    api = HfApi()
    user = api.whoami()['name']
    
    default_name = "ai-voice-detection-multilingual"
    model_name = input(f"Enter model name [default: {default_name}]: ").strip() or default_name
    
    repo_id = f"{user}/{model_name}"
    
    # 3. Create Repo
    print(f"\nCreating repository: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # 4. Upload
    print(f"Uploading files from {MODEL_PATH}...")
    try:
        api.upload_folder(
            folder_path=str(MODEL_PATH),
            repo_id=repo_id,
            repo_type="model"
        )
        print("\n" + "="*60)
        print(f"SUCCESS! Model is live at: https://huggingface.co/{repo_id}")
        print("="*60)
        
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    upload_model()
