import requests
import sys
import json

url = "https://aozahran2025.pythonanywhere.com/classify"

image_path = sys.argv[1]

with open(image_path, "rb") as image_file:
    data = {"image": (image_file.name, image_file)}
    response = requests.post(url, files=data)
    if response.status_code == 200:
        data = json.loads(response.text)
        print(f"Class Index: {data['class_index']}")
        print(f"Class Name : {data['class_name']}")
    else:
        print(f"Error: {response.status_code}")