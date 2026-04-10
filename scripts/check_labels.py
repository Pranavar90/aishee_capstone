from src import data_loader
import os

path = os.path.join(os.getcwd(), 'data_download')
s, n = data_loader.get_file_paths(path)
print(f"Scream: {len(s)}")
print(f"Non-Scream: {len(n)}")
if len(n) > 0:
    print("Sample non-scream path:", n[0])
