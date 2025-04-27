from huggingface_hub import HfApi, HfFolder, upload_folder, snapshot_download
import torch
import os
import shutil

class Saver:
    def __init__(self):
        """
        Automatically uses the cached token from login.
        """
        self.api = HfApi()
        self.token = HfFolder.get_token()

    def push(self, model_path, logs_dir, repo_id, commit_message="Add model and logs"):
        """
        Uploads model and logs to Hugging Face Hub.

        Args:
            model_path (str): Path to the PyTorch model (.pt or .bin).
            logs_dir (str): Directory to scan for log files to upload.
            repo_id (str): Hugging Face repo id (e.g., "username/repo_name").
            commit_message (str): Commit message for upload.
        """
        tmp_dir = "tmp_upload"

        # Clean tmp dir if exists
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # Copy model
        shutil.copy(model_path, os.path.join(tmp_dir, os.path.basename(model_path)))

        # Copy logs directory if it exists
        if os.path.isdir(logs_dir):
            shutil.copytree(logs_dir, os.path.join(tmp_dir, "logs"))
        else:
            print(f"Warning: logs directory '{logs_dir}' not found. No logs will be uploaded.")

        # Upload folder to Hugging Face
        upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=self.token,
        )

        # Clean up
        shutil.rmtree(tmp_dir)

    def load_checkpoint(self, repo_id, filename, device="cpu", output_logs_dir="logs"):
        """
        Loads a model file and the associated logs folder from a Hugging Face repo.

        Args:
            repo_id (str): Hugging Face repo id.
            filename (str): Model file name inside repo (e.g., "model.pt").
            device (str): Device to map model to.
            output_logs_dir (str): Directory to save the logs folder locally.

        Returns:
            torch.nn.Module: The loaded model.
        """
        # Download snapshot
        local_dir = snapshot_download(repo_id, token=self.token)

        # Load model
        model_file = os.path.join(local_dir, filename)
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file {filename} not found in repo {repo_id}.")
        
        model = torch.load(model_file, map_location=device)

        # If logs are included in the repo, download them
        logs_dir = os.path.join(local_dir, "logs")
        if os.path.isdir(logs_dir):
            # Copy logs to specified output directory
            shutil.copytree(logs_dir, output_logs_dir)
            print(f"Logs have been downloaded to {output_logs_dir}")
        else:
            print(f"No logs found in the repo {repo_id}.")
        
        return model
