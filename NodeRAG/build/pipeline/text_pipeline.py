from typing import Dict,List
import pandas as pd
import asyncio
import os
import json

from ...config import NodeConfig
from ...LLM import LLM_message
from ...storage import storage
from ..component import Text_unit
from ...logging.error import clear_cache
from ...logging import info_timer

class text_pipline():
    def __init__(self, config:NodeConfig)-> None:
            
            self.config = config
            self.texts = self.load_texts()
            
        
    def load_texts(self) -> pd.DataFrame:
        
        texts = storage.load_parquet(self.config.text_path)
        return texts
    
    async def text_decomposition_pipline(self) -> None:
        
        async_task = []
        self.config.tracker.set(len(self.texts),'Text Decomposition')
        
        for index, row in self.texts.iterrows():
            text = Text_unit(row['context'],row['hash_id'],row['text_id'])
            async_task.append(text.text_decomposition(self.config))
        await asyncio.gather(*async_task)
        
            
    def increment(self) -> None:
        
        exist_hash_id = []
        
        with open(self.config.text_decomposition_path,'r',encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                exist_hash_id.append(line['hash_id'])
        self.texts = self.texts[~self.texts['hash_id'].isin(exist_hash_id)]
        
    async def rerun(self) -> None:
        
        self.texts = self.load_texts()
        
        with open(self.config.LLM_error_cache,'r',encoding='utf-8') as f:
            LLM_store = []
            for line in f:
                line = json.loads(line)
                LLM_store.append(line)
        
        clear_cache(self.config.LLM_error_cache)
        
        await self.rerun_request(LLM_store)
        self.config.tracker.close()
        await self.text_decomposition_pipline()
                
    async def rerun_request(self,LLM_store:List[Dict]) -> None:
        tasks = []
        
        self.config.tracker.set(len(LLM_store),'Rerun LLM on error cache of text decomposition pipeline')
        
        for store in LLM_store:
            input_data = store['input_data']
            store.pop('input_data')
            input_data.update({'response_format':self.config.prompt_manager.text_decomposition})    
            tasks.append(self.request_save(input_data,store,self.config))
        await asyncio.gather(*tasks)
    
    async def request_save(self,
                        input_data:LLM_message,
                        meta_data:Dict) -> None:
        
        response = await self.config.client(input_data,cache_path=self.config.LLM_error_cache,meta_data = meta_data)
        
        # --- START OF MODIFICATION ---
        
        # Validate the structure of the LLM response before saving
        if self.validate_llm_response(response):
            # If the format is correct, save it to the main decomposition file
            with open(self.config.text_decomposition_path,'a',encoding='utf-8') as f:
                await f.write(json.dumps(response)+'\n')
        else:
            # If the format is invalid, log it as an error for later review/rerun.
            # This prevents malformed data from reaching the Graph pipeline.
            self.config.console.print(f"[bold yellow]WARN: Invalid LLM response format for text_hash_id {meta_data.get('hash_id')}. Caching as error.[/bold yellow]")
            
            # Construct a payload consistent with other cached errors
            error_payload = {
                "error": "Invalid response format from LLM",
                "invalid_response": response,
                "input_data": input_data,
                **meta_data
            }
            with open(self.config.LLM_error_cache, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(error_payload) + '\n')
                
        # --- END OF MODIFICATION ---
        
        self.config.tracker.update()
        
    def check_error_cache(self) -> None:
        
        if os.path.exists(self.config.LLM_error_cache):
            num = 0
            
            with open(self.config.LLM_error_cache,'r',encoding='utf-8') as f:
                for line in f:
                    num += 1
                    
            if num > 0:
                self.config.console.print(f"[red]LLM Error Detected,There are {num} errors")
                self.config.console.print("[red]Please check the error log")
                self.config.console.print("[red]The error cache is named LLM_error.jsonl, stored in the cache folder")
                self.config.console.print("[red]Please fix the error and run the pipeline again")
                raise Exception("Error happened in text decomposition pipeline, Error cached.")

    @info_timer(message='Text Pipeline')
    async def main(self) -> None:
        
        if os.path.exists(self.config.text_decomposition_path):
            if os.path.getsize(self.config.text_decomposition_path) > 0:
                self.increment()
                
        await self.text_decomposition_pipline()
        self.config.tracker.close()
        self.check_error_cache()


    def validate_llm_response(self, response: Dict) -> bool:
        """
        Validates the structure of the LLM response to ensure it matches
        the format expected by the downstream Graph pipeline.

        Args:
            response (Dict): The full response object from the LLM client.

        Returns:
            bool: True if the format is valid, False otherwise.
        """
        # The expected data is nested under the 'response' key
        data = response.get('response')

        # 1. Check if the 'response' key exists and its value is a list
        if not isinstance(data, list):
            self.config.console.print(f"[bold yellow]Validation Failed: Top-level 'response' key is not a list. Found type {type(data)}.[/bold yellow]")
            return False

        # 2. An empty list is a valid response (text with no entities).
        if not data:
            return True

        # 3. Check each item within the list
        required_keys = {'semantic_unit', 'entities', 'relationships'}
        for index, item in enumerate(data):
            # 3a. Each item must be a dictionary
            if not isinstance(item, dict):
                self.config.console.print(f"[bold yellow]Validation Failed: Item at index {index} is not a dictionary.[/bold yellow]")
                return False

            # 3b. Each dictionary must contain the essential keys
            if not required_keys.issubset(item.keys()):
                missing_keys = required_keys - item.keys()
                self.config.console.print(f"[bold yellow]Validation Failed: Item at index {index} is missing keys: {missing_keys}[/bold yellow]")
                return False

        # If all checks pass, the format is valid
        return True
    