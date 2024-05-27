import requests
import json

class OllamaError(Exception):
    """Base exception class for OllamaClient errors."""
    pass

class ModelNotFound(OllamaError):
    """Raised when a requested model is not found."""
    pass

class APIRequestError(OllamaError):
    """Raised when there's an error with the Ollama API request."""
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434/api"):
        self.base_url = base_url

    def generate_text(self, model, prompt, stream=False, **kwargs):
        """Generates text using the specified Ollama model."""
        return self._make_request("generate", model=model, prompt=prompt, stream=stream, **kwargs)

    def generate_chat(self, model, messages, stream=False, **kwargs):
        """Generates chat responses using the specified Ollama model."""
        return self._make_request("chat", model=model, messages=messages, stream=stream, **kwargs)

    def get_embeddings(self, model, prompt, **kwargs):
        """Generates embeddings for the given prompt using the specified model."""
        url = f"{self.base_url}/embeddings"
        data = {"model": model, "prompt": prompt, **kwargs}
        response = requests.post(url, json=data)
        self._check_response(response)
        return response.json()

    def list_models(self):
        """Retrieves a list of available models from the Ollama server."""
        url = f"{self.base_url}/tags"
        response = requests.get(url)
        self._check_response(response) 
        return response.json()['models']

    def pull_model(self, model, insecure=False, stream=False):
        """Downloads a model from the Ollama library."""
        url = f"{self.base_url}/pull"
        data = {"name": model, "insecure": insecure, "stream": stream}

        if stream:
            return self._stream_request(url, data)
        else:
            response = requests.post(url, json=data)
            self._check_response(response)
            return response.json()

    def _make_request(self, endpoint, model, stream=False, **kwargs):
        """Helper function to make requests to the Ollama API endpoints."""
        url = f"{self.base_url}/{endpoint}"
        data = {"model": model, "stream": stream, **kwargs}

        if stream:
            return self._stream_request(url, data)
        else:
            response = requests.post(url, json=data)
            self._check_response(response)  
            return response.json()

    def _stream_request(self, url, data):
        """Helper function to handle streaming requests."""
        with requests.post(url, json=data, stream=True) as response:
            self._check_response(response)
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise APIRequestError(f"Error decoding JSON: {e}", response=response) from e

    def _check_response(self, response):
        """Checks the API response for errors and raises exceptions accordingly."""
        if response.status_code == 404:
            raise ModelNotFound(f"Model not found: {response.text}")
        elif not response.ok:
            raise APIRequestError(f"API request failed with status code {response.status_code}: {response.text}", response=response) 