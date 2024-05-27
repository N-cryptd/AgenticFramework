from Clients.OllamaClient import OllamaClient, OllamaError, APIRequestError

client = OllamaClient()  # Use default URL

try:
    # List available models
    models = client.list_models()
    print("Available Models:")
    for model in models:
        print(model['name'])

    # Pull a model (example)
    client.pull_model(model="llama3") 
    print("Model 'llama3' pulled successfully!")

except OllamaError as e:
    print(f"Ollama Error: {e}")
    if isinstance(e, APIRequestError) and e.response:
        print(f"API Response: {e.response.text}")

response = client.generate_text(model="llama3", prompt="Tell me a joke.")
print(f"Response: {response['response']}")

messages = [{"role": "user", "content": "What is the meaning of life?"}]
response = client.generate_chat(model="llama3", messages=messages)
print(f"Response: {response['message']['content']}")

embeddings = client.get_embeddings(model="all-minilm", prompt="This is a test sentence.")
print(f"Embeddings: {embeddings}")