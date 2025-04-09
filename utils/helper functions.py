import subprocess

def cloneRepository(repositoryURL, destination):
    try:
        subprocess.run(["git", "clone", repositoryURL, destination], check=True)
        print(f"Repository cloned to {destination}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

