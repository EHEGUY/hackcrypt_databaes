"""from pathlib import Path
import requests

url = "http://0.0.0.0:8010/api/v1/analyze-video"
all_responses = []  # 1. Initialize as a list
path_to_search = Path('.')

# Recursively find all .mp4 files
for file_path in path_to_search.rglob('*.mp4'):
    # 2. Use 'with' to open files in binary read mode ('rb')
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'video/mp4')}
        response = requests.post(url, files=files)
        
        # 3. Check for successful response before parsing
        if response.status_code == 200:
            data = response.json()
            # 4. Use .append() if result is a dict, or .extend() if it's a list
            all_responses.append(data)
        else:
            print(f"Failed to analyze {file_path.name}: {response.status_code}")

# Print the combined list of results
print(all_responses)
with open('out.txt', 'w', encoding='utf-8') as f:
    # Option A: Save as formatted JSON (Recommended)
    #json.dump(all_responses, f, indent=4)
    
    # Option B: Save as a plain string (if you don't want JSON formatting)
     f.write(str(all_responses))

print("Results successfully saved to out.txt")"""
from ultralytics import YOLO

# Load the actual model file
model = YOLO('yolov8n-face.pt')