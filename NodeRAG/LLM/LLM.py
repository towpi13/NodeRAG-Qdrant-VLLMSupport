import os
import backoff
from ..utils.lazy_import import LazyImport
from json import JSONDecodeError
import json
import aiohttp
import requests

from ..logging.error import (
    error_handler,
    error_handler_async
)
from httpx import (
    RequestError,
    HTTPStatusError
)
import httpx


from ..LLM.LLM_base import (
    LLM_message,
    ModelConfig,
    LLMOutput,
    Embedding_message,
    Embedding_output,
    LLMBase,
    OpenAI_message,
    Gemini_content
)


from openai import (
    RateLimitError,
    Timeout,
    APIConnectionError,
)

from google.api_core.exceptions import (
    ResourceExhausted,
    TooManyRequests,
    InternalServerError
)





GEMINI_API_URL="https://generativelanguage.googleapis.com/v1beta/models"




OpenAI = LazyImport('openai','OpenAI')
AzureOpenAI = LazyImport('openai','AzureOpenAI')
AsyncOpenAI = LazyImport('openai','AsyncOpenAI')
AsyncAzureOpenAI = LazyImport('openai','AsyncAzureOpenAI')
genai = LazyImport("google.genai")
# Together = LazyImport('together','Together')
# AsyncTogether = LazyImport('together','AsyncTogether')
    

class LLM(LLMBase):
    
    def __init__(self,
                 model_name: str,
                 api_keys: str | None,
                 config: ModelConfig | None = None) -> None:

        super().__init__(model_name, api_keys, config)
        
    def extract_config(self, config: ModelConfig) -> ModelConfig:
        return config
        
    def predict(self, input: LLM_message) -> LLMOutput:
        response = self.API_client(input)
        return response

    

    async def predict_async(self, input: LLM_message) -> LLMOutput:
        response = await self.API_client_async(input)
        return response
    
    def API_client(self, input: LLM_message) -> LLMOutput:
        pass
    
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        pass
    
    
class OPENAI(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None=None) -> None:
        
        super().__init__(model_name, api_keys, Config)
        
        if self.api_keys is None:
            self.api_keys = os.getenv("OPENAI_API_KEY")
            
        self.client = OpenAI(api_key=self.api_keys)
        self.client_async = AsyncOpenAI(api_key=self.api_keys)
        self.config = self.extract_config(Config)
    
        
    def extract_config(self, config: ModelConfig) -> ModelConfig:
        options = {
            "max_tokens": config.get("max_tokens", 10000),  # Default value if not provided
            "temperature": config.get("temperature", 0.0),  # Default value if not provided
        }
        return options
    
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError,JSONDecodeError], 
                          max_time=30, 
                          max_tries=4)
    def _create_completion(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        
        if response_format:
            
            params["response_format"] = response_format
            response = self.client.beta.chat.completions.parse(**params)
            json_response = response.choices[0].message.parsed.model_dump_json()
            json_response = json.loads(json_response)

            return json_response

        else:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()

        
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError,JSONDecodeError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_completion_async(self, messages, response_format=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            **self.config
        }
        if response_format:
            params["response_format"] = response_format
            response = await self.client_async.beta.chat.completions.parse(**params)
            json_response = response.choices[0].message.parsed.model_dump_json()
            json_response = json.loads(json_response)
            return json_response
        else:

            response = await self.client_async.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        

    @error_handler
    def API_client(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = self._create_completion(
            messages, 
            input.get('response_format')
        )
        return response

    @error_handler_async
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = await self._create_completion_async(
            messages, 
            input.get('response_format')
        )
        
        return response
    
    def stream_chat(self,input:LLM_message):
        messages = self.messages(input)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def messages(self, input: LLM_message) -> OpenAI_message:
        
        messages = []
        if input.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": input["system_prompt"]
            })
        content =[{"type": "text","text": input["query"]}]
        
        messages.append({"role": "user","content": content})
        
        return messages
    

class OpenAI_Embedding(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys,Config)
        
        if api_keys is None:
            api_keys = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_keys)
        self.client_async = AsyncOpenAI(api_key=api_keys)
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    def _create_embedding(self, input: Embedding_message) -> Embedding_output:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [res.embedding for res in response.data]
    
    @error_handler
    def API_client(self, input: Embedding_message) -> Embedding_output:
        response = self._create_embedding(input)
        
        return response
    
    @backoff.on_exception(backoff.expo, 
                          [RateLimitError, Timeout, APIConnectionError], 
                          max_time=30, 
                          max_tries=4)
    async def _create_embedding_async(self, input: Embedding_message) -> Embedding_output:
        response = await self.client_async.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [res.embedding for res in response.data]
    
    @error_handler_async
    async def API_client_async(self, input: Embedding_message) -> Embedding_output:
        response = await self._create_embedding_async(input)
        return response
    
    
    
    
    

    

class Gemini(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys, Config)
        if self.api_keys is None:
            self.api_keys = os.getenv('GOOGLE_API_KEY')
        
        self.config = self.extract_config(Config)
        self.api_url = f"{GEMINI_API_URL}/{self.model_name}:generateContent"
        self.headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_keys}

    def extract_config(self, config: ModelConfig) -> ModelConfig:
        options = {
            "max_output_tokens": config.get("max_tokens", 8192),
            "temperature": config.get("temperature", 0.2),
        }
        return options

    def _prepare_payload(self, messages, response_format=None):
        payload = {
            "contents": [{"parts": [{"text": msg}]} for msg in messages],
            "generationConfig": {
                "temperature": self.config["temperature"],
                "maxOutputTokens": self.config["max_output_tokens"]
            }
        }
        if response_format:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        return payload

    @backoff.on_exception(backoff.expo, 
                          (requests.exceptions.RequestException, JSONDecodeError),
                          max_time=30, 
                          max_tries=4)
    def _create_completion(self, messages, response_format=None):
        payload = self._prepare_payload(messages, response_format)
        
        with requests.post(self.api_url, headers=self.headers, json=payload, timeout=180) as response:
            if response.status_code == 200:
                response_json = response.json()
                
                if "candidates" not in response_json or not response_json["candidates"]:
                    raise ValueError("API response is missing 'candidates'. Cannot process.")

                json_text_from_api = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
                if response_format:
                    return json.loads(json_text_from_api)
                return json_text_from_api
            else:
                response.raise_for_status()

    @backoff.on_exception(backoff.expo, 
                          (aiohttp.ClientError, JSONDecodeError),
                          max_time=30, 
                          max_tries=4)
    async def _create_completion_async(self, messages, response_format=None):
        payload = self._prepare_payload(messages, response_format)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=180
            ) as response:
                if response.status == 200:
                    response_json = await response.json()
                    
                    if "candidates" not in response_json or not response_json["candidates"]:
                        raise ValueError("API response is missing 'candidates'. Cannot process.")

                    json_text_from_api = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    
                    if response_format:
                        return json.loads(json_text_from_api)
                    return json_text_from_api
                else:
                    response.raise_for_status()

    @error_handler
    def API_client(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = self._create_completion(
            messages,
            input.get('response_format')
        )
        return response

    @error_handler_async
    async def API_client_async(self, input: LLM_message) -> LLMOutput:
        messages = self.messages(input)
        response = await self._create_completion_async(
            messages,
            input.get('response_format')
        )
        return response
    
    def messages(self, input: LLM_message) -> Gemini_content:
        query = ''
        if input.get("system_prompt"):
            query = 'system_prompt:\n'+input["system_prompt"]
        query = query + '\nquery:\n'+input["query"]
        content = [query]
        return content

    

class Gemini_Embedding(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        # Initialize using the parent LLM class
        super().__init__(model_name, api_keys, Config)
        if self.api_keys is None:
            self.api_keys = os.getenv('GOOGLE_API_KEY')
        
        # Set the specific endpoint for batch embeddings
        self.api_url = f"{GEMINI_API_URL}/{self.model_name}:batchEmbedContents"
        self.headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_keys}

    def extract_config(self, config: ModelConfig) -> ModelConfig:
        # This method is required by the abstract parent class.
        # Embedding models typically do not require extra configuration.
        return {}

    @backoff.on_exception(backoff.expo, 
                          (requests.exceptions.RequestException, JSONDecodeError), 
                          max_time=30, 
                          max_tries=4)
    def _create_embedding(self, input: Embedding_message) -> Embedding_output:
        # Construct the payload for the batchEmbedContents endpoint
        payload = {
            "requests": [
                {"model": f"models/{self.model_name}", "content": {"parts": [{"text": text}]}} 
                for text in input
            ]
        }
        import traceback
        try:
            with requests.post(self.api_url, headers=self.headers, json=payload, timeout=180) as response:
                response.raise_for_status()
                response_json = response.json()
                print("-----rESPONSE JSON-----", response_json)
                # Extract the vector values from the embeddings list
                return [embedding['values'] for embedding in response_json.get('embeddings', [])]

        except Exception as e:
            self.console.print('[bold red]Error in Gemini Embedding API call:[/bold red]', e)
            self.console.print(traceback.format_exc())
            raise e
        
    @error_handler
    def API_client(self, input: Embedding_message) -> Embedding_output:
        response = self._create_embedding(input)
        return response
    
    @backoff.on_exception(backoff.expo, 
                          (aiohttp.ClientError, JSONDecodeError),
                          max_time=30, 
                          max_tries=4)
    async def _create_embedding_async(self, input: Embedding_message) -> Embedding_output:
        # Construct the payload for the batchEmbedContents endpoint
        payload = {
            "requests": [
                {"model": f"models/{self.model_name}", "content": {"parts": [{"text": text}]}} 
                for text in input
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=180
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                # Extract the vector values from the embeddings list
                return [embedding['values'] for embedding in response_json.get('embeddings', [])]

    @error_handler_async
    async def API_client_async(self, input: Embedding_message) -> Embedding_output:
        response = await self._create_embedding_async(input)
        return response
    


class VLLM_Embedding(LLM):
    def __init__(self, 
                 model_name: str, # This is the model name from the config
                 api_keys: str | None,
                 Config: dict | None) -> None:
        
        # MODIFIED: The primary source of truth is now the Config dictionary.
        # Call the parent constructor with the model_name from the config.
        super().__init__(model_name, api_keys, Config)
        
        if not Config:
            raise ValueError("A 'Config' dictionary is required for the vLLM provider.")
        
        # NEW: Explicitly get the base_url from the 'Config' dictionary.
        self.base_url = Config.get('base_url')
        if not self.base_url:
            raise ValueError("The 'Config' dictionary must contain a 'base_url' key for the vLLM provider.")
        
        # NEW: The model name for the payload is the primary 'model_name' from the config.
        # This instance variable is used by the _create_embedding methods.
        self.payload_model_name = model_name
        if not self.payload_model_name:
             raise ValueError("The 'Config' dictionary must contain a 'model_name' key.")

        # --- No changes below this line ---
        self.client = httpx.Client()
        self.client_async = httpx.AsyncClient()
        # The endpoint is constructed from the base_url read from the config.
        self.endpoint = f"{self.base_url.strip().rstrip('/')}/v1/embeddings"
    
    @backoff.on_exception(backoff.expo, 
                          (RequestError, HTTPStatusError), 
                          max_time=30, 
                          max_tries=4)


    def _create_embedding(self, input: Embedding_message) -> Embedding_output:
        # CHANGED: The 'model' in the payload now uses the name from the Config.
        payload = {
            "model": self.payload_model_name,
            "input": input
        }
        response = self.client.post(self.endpoint, json=payload, timeout=60.0)
        response.raise_for_status()
        response_data = response.json()
        return [res['embedding'] for res in response_data['data']]
    
    @error_handler
    def API_client(self, input: Embedding_message) -> Embedding_output:
        response = self._create_embedding(input)
        return response
    
    @backoff.on_exception(backoff.expo, 
                          (RequestError, HTTPStatusError), 
                          max_time=30, 
                          max_tries=4)
    async def _create_embedding_async(self, input: Embedding_message) -> Embedding_output:
        # CHANGED: The 'model' in the payload now uses the name from the Config.
        payload = {
            "model": self.payload_model_name,
            "input": input
        }
        response = await self.client_async.post(self.endpoint, json=payload, timeout=60.0)
        response.raise_for_status()
        response_data = response.json()
        return [res['embedding'] for res in response_data['data']]
    
    @error_handler_async
    async def API_client_async(self, input: Embedding_message) -> Embedding_output:
        response = await self._create_embedding_async(input)
        return response