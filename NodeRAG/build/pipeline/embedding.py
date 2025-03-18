from typing import Dict
import os
import asyncio
import json
import math


from ...config import NodeConfig
from ...LLM import Embedding_message


from ...storage import (
    Mapper,
    storage
)

from ...logging import info_timer

class Embedding_pipeline():

    def __init__(self,config:NodeConfig):
        self.config = config
        self.embedding_client = self.config.embedding_client
        self.mapper = self.load_mapper()
        
        

    def load_mapper(self) -> Mapper:
        mapping_list = [self.config.text_path,
                        self.config.semantic_units_path,
                        self.config.attributes_path]
        mapping_list = [path for path in mapping_list if os.path.exists(path)]
        return Mapper(mapping_list)
    
    async def get_embeddings(self,context_dict:Dict[str,Embedding_message]):
        
        empty_ids = [key for key, value in context_dict.items() if value == ""]
        
        if len(empty_ids) > 0:
            
            context_dict = {key: value for key, value in context_dict.items() if value != ""}
            
            for empty_id in empty_ids:
                self.mapper.delete(empty_id)

        
        embedding_input = list(context_dict.values())
        
        ids = list(context_dict.keys())
        
        embedding_output = await self.embedding_client(embedding_input,cache_path=self.config.LLM_error_cache,meta_data = {'ids':ids})
        
        if embedding_output == 'Error cached':
            return

        
        with open(self.config.embedding_cache,'a',encoding='utf-8') as f:

            for i in range(len(ids)):
                line = {'hash_id':ids[i],'embedding':embedding_output[i]} 
                f.write(json.dumps(line)+'\n')
                
        self.config.tracker.update()
    
    def delete_embedding_cache(self):
        
        if os.path.exists(self.config.embedding_cache):
            os.remove(self.config.embedding_cache)
    
            
            
    async def generate_embeddings(self):
        tasks = []
        none_embedding_ids = self.mapper.find_none_embeddings()
        self.config.tracker.set(math.ceil(len(none_embedding_ids)/self.config.embedding_batch_size),desc='Generating embeddings')
        for i in range(0,len(none_embedding_ids),self.config.embedding_batch_size):
            context_dict = {}
            for id in none_embedding_ids[i:i+self.config.embedding_batch_size]:
                context_dict[id] = self.mapper.get(id,'context')
            tasks.append(self.get_embeddings(context_dict))
        await asyncio.gather(*tasks)
        self.config.tracker.close()
        
    def insert_embeddings(self):
        
        if not os.path.exists(self.config.embedding_cache):
            return None
        
        with open(self.config.embedding_cache,'r',encoding='utf-8') as f:
            lines = []
            for line in f:
                line = json.loads(line.strip())
                if isinstance(line['embedding'],str):
                    continue
                self.mapper.add_attribute(line['hash_id'],'embedding','done')
                lines.append(line)
        
        storage(lines).save_parquet(self.config.embedding,append=os.path.exists(self.config.embedding))
        self.mapper.update_save()
        
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
                    raise Exception("Error happened in embedding pipeline, Error cached.")
                    
    async def rerun(self):
        
        with open(self.config.LLM_error_cache,'r',encoding='utf-8') as f:
            LLM_store = []
            
            for line in f:
                line = json.loads(line)
                LLM_store.append(line)
        
        tasks = []
        context_dict = {}
        
        self.config.tracker.set(len(LLM_store),desc='Rerun embedding')
        
        for store in LLM_store:
            input_data = store['input_data']
            meta_data = store['meta_data']
            store.pop('input_data')
            store.pop('meta_data')
            tasks.append(self.request_save(input_data,store,self.config))
        
        await asyncio.gather(*tasks)
        self.config.tracker.close()
        self.insert_embeddings()
        self.delete_embedding_cache()
        self.check_error_cache()
        await self.main_async()
        
    async def request_save(self,
                           input_data:Embedding_message,
                           meta_data:Dict,
                           config:NodeConfig) -> None:
        
        response = await config.client(input_data,cache_path=config.LLM_error_cache,meta_data = meta_data)
        
        if response == 'Error cached':
            return
        
        with open(self.config.embedding_cache,'a',encoding='utf-8') as f:
            for i in range(len(meta_data['ids'])):
                line = {'hash_id':meta_data['ids'][i],'embedding':response[i]} 
                f.write(json.dumps(line)+'\n')

    

    def check_embedding_cache(self):
        if os.path.exists(self.config.embedding_cache):
            self.insert_embeddings()
            self.delete_embedding_cache()
            
    @info_timer(message='Embedding Pipeline')
    async def main(self):
        self.check_embedding_cache()
        await self.generate_embeddings()
        self.insert_embeddings()
        self.delete_embedding_cache()    
        self.check_error_cache()
            
        
    
    
    
    