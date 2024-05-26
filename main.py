from Clients.OllamaClient import OllamaClient

client = OllamaClient()  # Use default URL

response = client.generate_text(model="llama3", prompt="Tell me a joke.")
print(f"Response: {response['response']}")

messages = [{"role": "user", "content": "What is the meaning of life?"}]
response = client.generate_chat(model="llama3", messages=messages)
print(f"Response: {response['message']['content']}")

embeddings = client.get_embeddings(model="all-minilm", prompt="This is a test sentence.")
print(f"Embeddings: {embeddings}")