from typing import Dict
import os
import asyncio
import json
import math
import uuid


from ...config import NodeConfig
from ...LLM import Embedding_message


from ...storage import (
    Mapper,
    storage
)

from ...logging import info_timer

from qdrant_client import AsyncQdrantClient, models

class Embedding_pipeline():
    def __init__(self, config: NodeConfig):
        self.config = config
        self.embedding_client = self.config.embedding_client
        self.mapper = self.load_mapper()
        
        self.qdrant_client = None # Default to None
        
        if getattr(self.config, 'vector_store', None) == 'qdrant':
            self.config.console.print("[bold cyan]Embedding pipeline is in Qdrant mode.[/bold cyan]")
            qdrant_url = getattr(self.config, 'qdrant_url', None)
            if not qdrant_url:
                raise ValueError("Qdrant config requires a 'qdrant_url'.")

            self.qdrant_client = AsyncQdrantClient(
                url=qdrant_url, 
                api_key=getattr(self.config, 'qdrant_api_key', None),
                timeout=30.0 
            )
        else:
            # Add a log to be 100% sure we are in the correct mode
            self.config.console.print("[bold cyan]Embedding pipeline is in Local File (HNSW) mode.[/bold cyan]")
        
        

    async def _ensure_qdrant_collection_exists_async(self):
        """
        Checks if the configured Qdrant collection exists, and creates it if it doesn't.
        This is an idempotent operation, safe to run every time.
        """
        collection_name = getattr(self.config, 'qdrant_collection_name', None)
        embedding_dim = getattr(self.config, 'embedding_dim', None) # Renamed from 'dim' for clarity

        # Use the 'dim' from the HNSW config as a fallback if 'embedding_dim' isn't defined
        if not embedding_dim:
            embedding_dim = getattr(self.config, 'dim', None)

        if not collection_name or not embedding_dim:
            raise ValueError("Qdrant config requires 'qdrant_collection_name' and 'embedding_dim' (or 'dim').")

        self.config.console.print(f"[cyan]Ensuring Qdrant collection '{collection_name}' exists...[/cyan]")
        
        try:
            # Check if the collection already exists
            await self.qdrant_client.get_collection(collection_name=collection_name)
            self.config.console.print(f"[green]Collection '{collection_name}' already exists.[/green]")
        
        except Exception as e:
            # An exception here likely means the collection does not exist.
            self.config.console.print(f"[yellow]Collection '{collection_name}' not found. Creating it now...[/yellow]")
            try:
                await self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim, 
                        distance=models.Distance.COSINE # Or another distance metric of your choice
                    ),
                )
                self.config.console.print(f"[bold green]Successfully created collection '{collection_name}'.[/bold green]")
            except Exception as create_e:
                self.config.console.print(f"[bold red]FATAL: Failed to create Qdrant collection '{collection_name}'. Error: {create_e}[/bold red]")
                raise create_e # Re-raise the exception to halt the pipeline

    def load_mapper(self) -> Mapper:
        mapping_list = [self.config.text_path,
                        self.config.semantic_units_path,
                        self.config.attributes_path]
        mapping_list = [path for path in mapping_list if os.path.exists(path)]
        return Mapper(mapping_list)
    
    
    async def get_embeddings(self, context_dict: Dict[str, Embedding_message]):
        
        empty_ids = [key for key, value in context_dict.items() if value == ""]
        
        if len(empty_ids) > 0:
            context_dict = {key: value for key, value in context_dict.items() if value != ""}
            for empty_id in empty_ids:
                self.mapper.delete(empty_id)

        embedding_input = list(context_dict.values())
        ids = list(context_dict.keys())
        
        # Step 1: Call the embedding client
        embedding_output = await self.embedding_client(embedding_input, cache_path=self.config.LLM_error_cache, meta_data={'ids': ids})
        
        # Step 2: Vigorously validate the output from the embedding client
        if embedding_output == 'Error cached':
            self.config.console.print("[yellow]WARN: An error was cached by the embedding client. Skipping this batch.[/yellow]")
            return

        if not isinstance(embedding_output, list):
            self.config.console.print(f"[bold red]FATAL ERROR: Embedding client returned a non-list object of type {type(embedding_output)}. Response: {embedding_output}[/bold red]")
            # In a real scenario, you might want to raise an exception here
            return

        if len(embedding_output) != len(ids):
            self.config.console.print(f"[bold red]FATAL ERROR: Mismatch between number of inputs ({len(ids)}) and number of embeddings received ({len(embedding_output)}).[/bold red]")
            return
            
        # Step 3: Write to the temporary cache file, but validate EACH item
        lines_written = 0
        with open(self.config.embedding_cache, 'a', encoding='utf-8') as f:
            for i in range(len(ids)):
                embedding_vector = embedding_output[i]
                
                # The most important check: ensure the embedding itself is a list (of floats)
                if isinstance(embedding_vector, list):
                    line = {'hash_id': ids[i], 'embedding': embedding_vector}
                    f.write(json.dumps(line) + '\n')
                    lines_written += 1
                else:
                    # This will catch cases where the API returns a mix of valid embeddings and error strings
                    self.config.console.print(f"[bold red]ERROR: Skipping invalid embedding vector for hash_id {ids[i]}. Vector was not a list. Got: {embedding_vector}[/bold red]")
        
        if lines_written > 0:
            self.config.console.print(f"[green]Successfully wrote {lines_written} valid embeddings to temporary cache.[/green]")
        
        # This will now only be called if there are valid embeddings
        self.config.tracker.update()
    
    
    def delete_embedding_cache(self):
        
        if os.path.exists(self.config.embedding_cache):
            os.remove(self.config.embedding_cache)
    
            
    async def generate_embeddings(self):
        # --- THIS IS THE FIX ---
        # Delete any old cache file to ensure a clean run.
        self.delete_embedding_cache()
        # --- END OF FIX ---

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
        

    async def insert_embeddings_async(self):
        
        # Step 1: Check if the temporary cache file from the previous step exists.
        if not os.path.exists(self.config.embedding_cache):
            self.config.console.print(f"[bold yellow]WARN in Embedding pipeline: Embedding cache file not found at '{self.config.embedding_cache}'. No new embeddings to process.[/bold yellow]")
            return None
        
        # Step 2: Read and validate all embeddings from the cache file.
        lines = []
        with open(self.config.embedding_cache, 'r', encoding='utf-8') as f:
            for line in f:
                line_data = json.loads(line.strip())
                if isinstance(line_data.get('embedding'), list):
                    self.mapper.add_attribute(line_data['hash_id'], 'embedding', 'done')
                    lines.append(line_data)
                else:
                    self.config.console.print(f"[bold red]ERROR: Skipping invalid embedding data in cache for hash_id {line_data.get('hash_id')}.[/bold red]")

        # Step 3: Check if there are any valid embeddings to save.
        if not lines:
            self.config.console.print("[bold yellow]WARN: No valid embeddings found in cache file. Skipping insertion step.[/bold yellow]")
            return

        # Step 4: Conditional Storage Logic - Save to the correct destination.
        if self.qdrant_client:
            self.config.console.print(f"[cyan]Saving {len(lines)} embeddings to Qdrant...[/cyan]")
            collection_name = getattr(self.config, 'qdrant_collection_name')

            points_to_upsert = []
            for item in lines:
                # Get all data associated with this hash_id from our mapper
                full_data_record = self.mapper.get(item['hash_id'])

                # Create a rich payload
                payload = {
                    "hash_id": item['hash_id'],
                    "context": full_data_record.get('context', ''),  # The original text!
                    "type": full_data_record.get('type', 'unknown')   # e.g., 'semantic_unit'
                }
                
                # Create the point with the rich payload
                points_to_upsert.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, item['hash_id'])),
                        vector=item['embedding'], 
                        payload=payload
                    )
                )
            
            await self.qdrant_client.upsert(collection_name=collection_name, points=points_to_upsert, wait=True)
            self.config.console.print(f"[bold green]Successfully saved {len(lines)} embeddings to Qdrant.[/bold green]")
        
        else:
            self.config.console.print(f"[cyan]Saving {len(lines)} embeddings to local Parquet file: '{self.config.embedding}'...[/cyan]")
            storage(lines).save_parquet(self.config.embedding, append=os.path.exists(self.config.embedding))
            self.config.console.print(f"[bold green]Successfully saved embeddings to Parquet file.[/bold green]")
        
        # This runs for both cases
        self.mapper.update_save()
                

    def insert_embeddings(self):        
        # Check 1: Does the temporary cache file from the previous step even exist?
        if not os.path.exists(self.config.embedding_cache):
            self.config.console.print("[bold yellow]WARN: Embedding cache file not found. Skipping insertion. No embedding.parquet will be created.[/bold yellow]")
            return None
        
        lines = []
        with open(self.config.embedding_cache, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                # Check 2: Is the 'embedding' field a valid list of numbers?
                if not isinstance(line.get('embedding'), list):
                    self.config.console.print(f"[bold red]ERROR: Skipping invalid embedding data for hash_id {line.get('hash_id')}. Found a non-list value.[/bold red]")
                    continue # Skip this corrupted line
                
                self.mapper.add_attribute(line['hash_id'], 'embedding', 'done')
                lines.append(line)
        
        # Check 3: After reading and validating, did we find any valid embeddings to save?
        if not lines:
            self.config.console.print("[bold yellow]WARN: No valid embeddings found in the cache file to insert. The embedding.parquet file will not be created.[/bold yellow]")
            return

        # If we have valid lines, save them to the final destination.
        self.config.console.print(f"[green]Saving {len(lines)} valid embeddings to {self.config.embedding}...[/green]")
        storage(lines).save_parquet(self.config.embedding, append=os.path.exists(self.config.embedding))
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
        if self.qdrant_client:
            await self._ensure_qdrant_collection_exists_async()
            
        # It will only run if there are new embeddings to generate or cache to process.
        
        # --- THIS IS THE FIX ---
        # self.check_embedding_cache() # <--- REMOVE THIS LINE
        # --- END OF FIX ---
        
        await self.generate_embeddings()
        await self.insert_embeddings_async()
        self.delete_embedding_cache()     
        self.check_error_cache()
        
    
    
    
    