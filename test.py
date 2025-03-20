import requests

url = "http://127.0.0.1:5000/predict"  # Use localhost
data = {"product_id": "B00009W3HD"}

response = requests.post(url, json=data)
print(response.json())  # Should return trustworthiness score + feature ratings
