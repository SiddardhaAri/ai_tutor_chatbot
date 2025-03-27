import requests

headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

payload = {
    "model": "mistralai/mistral-7b-instruct:free",
    "messages": [{"role": "user", "content": "What is AI?"}]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.text)
