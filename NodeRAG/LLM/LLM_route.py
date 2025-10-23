import asyncio
from .LLM_base import I,O,Dict
from .LLM import *

from ..logging.error import (
    cache_error,
    cache_error_async
)

from rich.console import Console
from typing import Optional

def LLM_route(config : ModelConfig, console: Optional[Console] = None) -> LLM: # <-- MODIFIED 2: Accept console
    
    '''Route the request to the appropriate LLM service provider'''
        
    # --- MODIFIED 3: Pass console to ALL constructors ---
    # This ensures that if any of them need logging, they have it.
    
    service_provider = config.get("service_provider")
    model_name = config.get("model_name")
    embedding_model_name = config.get("model_name",None)
    api_keys = config.get("api_keys",None)
        
    match service_provider:
        case "openai":
            return OPENAI(model_name, api_keys, config) # We'll pass console here for consistency
        case "openai_embedding":
            return OpenAI_Embedding(embedding_model_name, api_keys, config)
        case "gemini":
            return Gemini(model_name, api_keys, config)
        case "gemini_embedding":
            # This is the one that truly needs it
            return Gemini_Embedding(embedding_model_name, api_keys, config, console=console) 
        case "vllm":
            if not embedding_model_name:
                raise ValueError("For 'vllm' provider, the 'embedding_model_name' must be set to the service URL.")
            return VLLM_Embedding(embedding_model_name, api_keys, config)
            
        case _:
            raise ValueError(f"Service provider '{service_provider}' not supported")
   
            

class API_client():
    def __init__(
        self, 
        config : ModelConfig,
        console: Optional[Console] = None) -> None: # <-- Accept console here
        
        # Pass console to the routing function
        self.llm = LLM_route(config, console=console) # <-- Pass it here
        self.rate_limit = config.get("rate_limit",50)
        self.semaphore = asyncio.Semaphore(self.rate_limit)
        


            
    @cache_error_async
    async def __call__(self, input: I, *,cache_path:str|None=None,meta_data:Dict|None=None) -> O:
        
        async with self.semaphore:
            response = await self.llm.predict_async(input)

            
        return response
    
    @cache_error
    def request(self, input:I, *,cache_path:str|None=None,meta_data:Dict|None=None) -> O:
        

        response = self.llm.predict(input)
        
        
        return response
    
    def stream_chat(self,input:I):
        yield from self.llm.stream_chat(input)
    
    
