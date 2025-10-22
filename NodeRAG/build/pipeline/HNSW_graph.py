import os
from ...utils.HNSW import HNSW
from ...storage import Mapper
from ...config import NodeConfig
from ...logging import info_timer



class HNSW_pipeline():
    
    def __init__(self,config:NodeConfig):
        
        self.config = config
        
        if not os.path.exists(self.config.embedding):
            self.config.console.print(f"[bold red]FATAL ERROR in HNSW Pipeline: The required embedding file is missing.[/bold red]")
            self.config.console.print(f"Path not found: '{self.config.embedding}'")
            self.config.console.print("[bold red]This means the Embedding pipeline failed to produce its output. Cannot continue.[/bold red]")
            # Raise a clear exception to halt the entire process.
            raise FileNotFoundError(f"Cannot start HNSW pipeline, required embedding file not found: {self.config.embedding}")
        
        self.mapper = self.load_mapper()
        self.hnsw = self.load_hnsw()

    def load_mapper(self) -> Mapper:
        
        mapping_list = [self.config.semantic_units_path,
                        self.config.attributes_path,
                        self.config.high_level_elements_path,
                        self.config.text_path]
        
        valid_mapping_list = [path for path in mapping_list if os.path.exists(path)]
        
        mapper = Mapper(valid_mapping_list)
        
        try:
            self.config.console.print(f"[cyan]Loading embeddings from '{self.config.embedding}'...[/cyan]")
            mapper.add_embedding(self.config.embedding)
            self.config.console.print(f"[green]Successfully loaded {len(mapper.embeddings)} embeddings.[/green]")
        
        except ValueError as e:
            self.config.console.print(f"[bold red]FATAL ERROR: Failed to process the embedding file. It appears to be corrupted.[/bold red]")
            self.config.console.print(f"File path: '{self.config.embedding}'")
            self.config.console.print(f"Error details: {e}")
            raise # Re-raise the exception to halt the pipeline
        except Exception as e:
            # Catch any other unexpected errors during file loading.
            self.config.console.print(f"[bold red]An unexpected error occurred while loading the embedding file: {e}[/bold red]")
            raise
            
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
        
        
            
        
    
        
