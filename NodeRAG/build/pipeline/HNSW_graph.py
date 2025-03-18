import os
from ...utils.HNSW import HNSW
from ...storage import Mapper
from ...config import NodeConfig
from ...logging import info_timer



class HNSW_pipeline():
    
    def __init__(self,config:NodeConfig):
        
        self.config = config
        self.mapper = self.load_mapper()
        self.hnsw = self.load_hnsw()
        
    def load_mapper(self) -> Mapper:
        
        mapping_list = [self.config.semantic_units_path,
                        self.config.attributes_path,
                        self.config.high_level_elements_path,
                        self.config.text_path]
        
        for i in range(len(mapping_list)):
            if not os.path.exists(mapping_list[i]):
                mapping_list.pop(i)
        
        mapper = Mapper(mapping_list)
        if os.path.exists(self.config.embedding):
            mapper.add_embedding(self.config.embedding)
            
        return mapper
    
    def load_hnsw(self) -> HNSW:
        
        hnsw = HNSW(self.config)
        
        if os.path.exists(self.config.HNSW_path):
            
            hnsw.load_HNSW(self.config.HNSW_path)
            return hnsw
        
        elif self.mapper.embeddings is not None:
            return hnsw
        else:
            raise Exception('No embeddings found')

    
    def generate_HNSW(self):
        unHNSW = self.mapper.find_non_HNSW()
        
        self.config.console.print(f'[yellow]Generating HNSW graph for {len(unHNSW)} nodes[/yellow]')
        self.hnsw.add_nodes(unHNSW)
        self.config.console.print(f'[green]HNSW graph has been added to the graph[/green]')
        self.config.tracker.set(len(unHNSW),desc="storing HNSW graph")
        for id,embedding in unHNSW:
            self.mapper.add_attribute(id,'embedding','HNSW')
            self.config.tracker.update()
        self.config.tracker.close()
        self.config.console.print(f'[green]HNSW graph generated for {len(unHNSW)} nodes[/green]')
    
    def delete_embedding(self):
        
        if os.path.exists(self.config.embedding):
            os.remove(self.config.embedding)
    
    @info_timer(message='HNSW graph generation')
    async def main(self):
        if os.path.exists(self.config.embedding):
            self.generate_HNSW()
            self.hnsw.save_HNSW()
            self.mapper.update_save()
            self.delete_embedding()
            self.config.console.print('[green]HNSW graph saved[/green]')
        
        
            
        
    
        
