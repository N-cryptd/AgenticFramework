import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api"):
        self.base_url = base_url

    def generate_text(self, model, prompt, stream=False, **kwargs):
        """
        Generates text using the specified Ollama model.

        Args:
            model (str): The name of the Ollama model.
            prompt (str): The prompt to send to the model.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional parameters for the /api/generate endpoint.

        Returns:
            Union[dict, Generator[dict, None, None]]: 
                A dictionary containing the response if `stream=False`, otherwise a generator yielding response chunks.
        """
        url = f"{self.base_url}/generate"
        data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}
        response = requests.post(url, json=data, stream=stream)
        
        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def generate_chat(self, model, messages, stream=False, **kwargs):
        """
        Generates chat responses using the specified Ollama model.

        Args:
            model (str): The name of the Ollama model.
            messages (list): A list of chat messages in the format specified by the Ollama API.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional parameters for the /api/chat endpoint.

        Returns:
            Union[dict, Generator[dict, None, None]]: 
                A dictionary containing the response if `stream=False`, otherwise a generator yielding response chunks.
        """
        url = f"{self.base_url}/chat"
        data = {"model": model, "messages": messages, "stream": stream, **kwargs}
        response = requests.post(url, json=data, stream=stream)

        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    def get_embeddings(self, model, prompt, **kwargs):
        """
        Generates embeddings for the given prompt using the specified model.

        Args:
            model (str): The name of the Ollama model.
            prompt (str): The text to generate embeddings for.
            **kwargs: Additional parameters for the /api/embeddings endpoint.

        Returns:
            dict: A dictionary containing the embedding vector.
        """
        url = f"{self.base_url}/embeddings"
        data = {"model": model, "prompt": prompt, **kwargs}
        response = requests.post(url, json=data)
        return response.json() 

    def _stream_response(self, response):
        """Helper function to stream responses from the API."""
        for line in response.iter_lines():
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")