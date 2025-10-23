import json


from .unit import Unit_base
from ...storage import genid
from ...utils.readable_index import text_unit_index




text_unit_index_counter = text_unit_index()


class Text_unit(Unit_base):
    def __init__(self, raw_context:str = None,hash_id:str = None,human_readable_id:int = None,semantic_units:list = []):
        self.raw_context = raw_context
        self._hash_id = hash_id
        self._human_readable_id = human_readable_id
        
    @property
    def hash_id(self):
        if not self._hash_id:
            self._hash_id = genid([self.raw_context],"sha256")
        return self._hash_id
    
    @property
    def human_readable_id(self):
        if not self._human_readable_id:
            self._human_readable_id = text_unit_index_counter.increment()
        return self._human_readable_id    

        
 
    
    async def text_decomposition(self,config) -> dict: # Modified to return a dictionary
                
        prompt = config.prompt_manager.text_decomposition.format(text=self.raw_context)
        json_format = config.prompt_manager.text_decomposition_json
        input_data = {'query':prompt,'response_format':json_format}
        meta_data = {'text_hash_id':self.hash_id,'text_id':self.human_readable_id}

        response = await config.API_client(input_data,cache_path =config.LLM_error_cache,meta_data = meta_data)
        
        config.tracker.update()

        if response == 'Error cached':
            return None # Return None on error
        
        # Instead of writing to a file, create the data packet and return it.
        data = {**meta_data,'response':response}

        if config.use_local_cache:
            cache_path = config.text_decomposition_path
            with open(cache_path, 'a',encoding='utf-8') as f:
                f.write(json.dumps(data,ensure_ascii=False)+'\n')
                
        return data
        
    
