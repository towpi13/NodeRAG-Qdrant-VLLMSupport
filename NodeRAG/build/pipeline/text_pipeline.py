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
            
            with open(self.config.text_decomposition_path,'a',encoding='utf-8') as f:
                await f.write(json.dumps(response)+'\n')
            
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

            
                
                
        
            
            
            
    