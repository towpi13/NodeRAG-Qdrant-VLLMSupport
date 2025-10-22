import leidenalg as la
import os
import json
import asyncio
import faiss
import math
import numpy as np
from qdrant_client import AsyncQdrantClient
import uuid

from ...storage import (
    Mapper,
    storage
)

from ..component import (
    Community_summary,
    High_level_elements
)
from ...config import NodeConfig


from ...utils import (
    IGraph,
)

from ...logging import info_timer

class SummaryGeneration:
    def __init__(self,config:NodeConfig):
        
        self.config = config
        self.indices = self.config.indices
        self.communities = []
        self.high_level_elements = []
        self.qdrant_client = None 

        if os.path.exists(self.config.graph_path):
            self.mapper = Mapper([self.config.semantic_units_path,
                                    self.config.attributes_path])
            self.G = storage.load(self.config.graph_path)
            self.G_ig = IGraph(self.G).to_igraph()
            self.nodes_high_level_elements_group = []
            self.nodes_high_level_elements_match = []

            if getattr(self.config, 'vector_store', None) == 'qdrant':
                self.config.console.print("[bold cyan]Summary pipeline is in Qdrant mode. Will fetch embeddings from the server.[/bold cyan]")
                self.qdrant_client = AsyncQdrantClient(
                    url=getattr(self.config, 'qdrant_url'),
                    api_key=getattr(self.config, 'qdrant_api_key', None)
                )
            else:
                self.config.console.print("[bold cyan]Summary pipeline is in local file mode. Loading embeddings from disk.[/bold cyan]")
                if os.path.exists(self.config.embedding):
                    self.mapper.add_embedding(self.config.embedding)
                else:
                    self.config.console.print(f"[bold red]ERROR: Local embedding file not found at {self.config.embedding}. Summary generation may fail.[/bold red]")

            
    def partition(self):
        
        partition = la.find_partition(self.G_ig,la.ModularityVertexPartition)
        
        for i,community in enumerate(partition):
            community_name = [self.G_ig.vs[node]['name'] for node in community if self.G_ig.vs[node]['name'] in self.mapper.mapping]
            
            self.communities.append(Community_summary(community_name,self.mapper,self.G,self.config))
            
    async def generate_community_summary(self,community:Community_summary):
        
        await community.generate_community_summary()
        if isinstance(community.response,str):
            self.config.tracker.update()
            return
        
        community_dict = {'community':community.community_node,
                          'response':community.response,
                          'hash_id':community.hash_id,
                          'human_readable_id':community.human_readable_id}
        
        with open(self.config.summary_path,'a',encoding='utf-8') as f:
            f.write(json.dumps(community_dict,ensure_ascii=False)+'\n')
        
        self.config.tracker.update()
        
        
            
    async def generate_high_level_element_summary(self):
        
        self.partition()
        
        tasks = []
        
        self.config.tracker.set(len(self.communities),'Community Summary')
        for community in self.communities:
            tasks.append(self.generate_community_summary(community))
        
        await asyncio.gather(*tasks)
        
        self.config.tracker.close()
       
        
    async def get_summary_embedding(self):
        tasks = []
        self.config.tracker.set(math.ceil(len(self.high_level_elements)/self.config.embedding_batch_size),'High Level Element Embedding')
        
        for i in range(0,len(self.high_level_elements),self.config.embedding_batch_size):
            high_level_element_batch = self.high_level_elements[i:i+self.config.embedding_batch_size]
            tasks.append(self.embedding_store(high_level_element_batch))
        await asyncio.gather(*tasks)
        self.config.tracker.close()
        
    async def embedding_store(self,high_level_element_batch:list[High_level_elements]):
        
        context = [high_level_element.context for high_level_element in high_level_element_batch]
        embedding = await self.config.embedding_client(context)
        
        for i in range(len(high_level_element_batch)):
            high_level_element_batch[i].store_embedding(embedding[i])
        self.config.tracker.update()

   
    async def high_level_element_summary(self):
        results = []
        
        # Safety Check 1: Ensure the summary file exists and is not empty before trying to read it.
        if not os.path.exists(self.config.summary_path) or os.path.getsize(self.config.summary_path) == 0:
            self.config.console.print("[bold yellow]WARN: Summary file is empty or does not exist. Skipping high-level element summary.[/bold yellow]")
            return # Exit the function gracefully

        with open(self.config.summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    self.config.console.print(f"[bold red]ERROR: Could not decode JSON from line: {line}[/bold red]")
                    continue
                
        All_nodes = []
        self.config.tracker.set(len(results),'High Level Element Summary')
        for result in results:
            high_level_elements = []
            node_names = result.get('community')
            response_data = result.get('response')

            # Safety Check 2: The 'response' must be a list to proceed.
            if isinstance(response_data, list):
                for high_level_element in response_data:
                    
                    # Safety Check 3: Ensure each item in the list is a dictionary with the keys we need.
                    if not isinstance(high_level_element, dict) or 'description' not in high_level_element or 'title' not in high_level_element:
                        self.config.console.print(f"[bold yellow]WARN: Skipping malformed high_level_element: {high_level_element}[/bold yellow]")
                        continue

                    he = High_level_elements(high_level_element['description'], high_level_element['title'], self.config)
                    he.related_node(node_names)
                    
                    if self.G.has_node(he.hash_id):
                        self.G.nodes[he.hash_id]['weight'] += 1
                        if self.G.has_node(he.title_hash_id):
                            self.G.nodes[he.title_hash_id]['weight'] += 1
                    else:
                        self.G.add_node(he.hash_id, type='high_level_element', weight=1)
                        self.G.add_node(he.title_hash_id, type='high_level_element_title', weight=1, related_node=he.hash_id)
                        high_level_elements.append(he)
                    
                    edge = (he.hash_id, he.title_hash_id)
                    
                    if not self.G.has_edge(*edge):
                        self.G.add_edge(*edge, weight=1)
            
            if node_names:
                All_nodes.extend(node_names)
            self.high_level_elements.extend(high_level_elements)
            self.config.tracker.update()

        self.config.tracker.close()

        # Safety Check 4: Don't proceed to embedding/clustering if no elements were created.
        if not self.high_level_elements:
            self.config.console.print("[bold yellow]WARN: No high-level elements were generated. Skipping embedding and clustering.[/bold yellow]")
            return
            
        await self.get_summary_embedding()

        centroids = math.ceil(math.sqrt(len(All_nodes) + len(self.high_level_elements)))
        threshold = (len(All_nodes) + len(self.high_level_elements)) / centroids if centroids > 0 else 0
        n = 0
        if threshold > self.config.Hcluster_size:
            embedding_list = None
            if self.qdrant_client:
                # QDRANT MODE: Fetch the vectors for the nodes from the Qdrant server.
                self.config.console.print(f"[cyan]Fetching {len(All_nodes)} vectors from Qdrant for clustering...[/cyan]")
                collection_name = getattr(self.config, 'qdrant_collection_name')
                
                # 1. We must convert the hash_ids in All_nodes to the UUIDs we used to store them.
                ids_to_retrieve = [str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id)) for node_id in All_nodes]
                
                # 2. The client call must be asynchronous with 'await'.
                retrieved_points = await self.qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=ids_to_retrieve,
                    with_vectors=True
                )
                
                # 3. We must map the results back correctly. The point.id is the UUID, but its
                #    payload contains the original hash_id we need.
                vector_map = {point.payload['hash_id']: point.vector for point in retrieved_points}
                embedding_list = np.array([vector_map[node_id] for node_id in All_nodes if node_id in vector_map], dtype=np.float32)

            else:
                # LOCAL FILE MODE: Original logic
                embedding_list = np.array([self.mapper.embeddings[node] for node in All_nodes], dtype=np.float32)

            high_level_element_embedding = np.array([he.embedding for he in self.high_level_elements], dtype=np.float32)
            
            # Ensure we actually got embeddings before trying to stack them
            if embedding_list is not None and embedding_list.size > 0:
                all_embeddings = np.vstack([high_level_element_embedding, embedding_list])

                kmeans = faiss.Kmeans(d=all_embeddings.shape[1], k=centroids)
                kmeans.train(all_embeddings.astype(np.float32))
                _, cluster_labels = kmeans.assign(all_embeddings.astype(np.float32))
                high_level_element_cluster_labels = cluster_labels[:len(self.high_level_elements)]
                embedding_cluster_labels = cluster_labels[len(self.high_level_elements):]
                self.config.console.print(f'[bold green]KMeans Clustering with {centroids} centroids[/bold green]')
            
                self.config.tracker.set(len(self.high_level_elements), 'Adding High Level Element Summary')
                for i in range(len(self.high_level_elements)):
                    for j in range(len(All_nodes)):
                        if high_level_element_cluster_labels[i] == embedding_cluster_labels[j] and All_nodes[j] in self.high_level_elements[i].related_node:
                            self.G.add_edge(All_nodes[j], self.high_level_elements[i].hash_id, weight=1)
                            n += 1
                    self.config.tracker.update()
            else:
                self.config.console.print("[bold yellow]WARN: No source embeddings found for clustering. Skipping edge creation based on clusters.[/bold yellow]")

        else:
            self.config.tracker.set(len(self.high_level_elements), 'Adding High Level Element Summary')
            for he in self.high_level_elements:
                for node in he.related_node:
                    self.G.add_edge(node, he.hash_id, weight=1)
                    n += 1
                self.config.tracker.update()
        
        self.config.tracker.close()
        self.config.console.print(f'[bold green]Added {n} edges[/bold green]')
                    
        
   
            
    
            
        
            
            

                
   
    def store_graph(self):
        storage(self.G).save_pickle(self.config.graph_path)
        self.config.console.print('[bold green]Graph stored[/bold green]')
        
    def delete_community_cache(self):
        os.remove(self.config.summary_path)
        
    def store_high_level_elements(self):
        
        high_level_elements = []
        titles = []
        embedding_list = []
        for high_level_element in self.high_level_elements:
            high_level_elements.append({'type':'high_level_element',
                                        'title_hash_id':high_level_element.title_hash_id,
                                        'context':high_level_element.context,
                                        'hash_id':high_level_element.hash_id,
                                        'human_readable_id':high_level_element.human_readable_id,
                                        'related_nodes':list(self.G.neighbors(high_level_element.hash_id)),
                                        'embedding':'done'})
            
            titles.append({'type':'high_level_element_title',
                           'hash_id':high_level_element.title_hash_id,
                           'context':high_level_element.title,
                           'human_readable_id':high_level_element.human_readable_id})
            
            embedding_list.append({'hash_id':high_level_element.hash_id,
                                   'embedding':high_level_element.embedding})
        G_high_level_elements = [node for node in self.G.nodes if self.G.nodes[node].get('type') == 'high_level_element']
        assert len(high_level_elements) == len(G_high_level_elements), f"The number of high level elements is not equal to the number of nodes in the graph. {len(high_level_elements)} != {len(G_high_level_elements)}"
        
        storage(high_level_elements).save_parquet(self.config.high_level_elements_path,append = os.path.exists(self.config.high_level_elements_path))
        storage(titles).save_parquet(self.config.high_level_elements_titles_path,append = os.path.exists(self.config.high_level_elements_titles_path))
        storage(embedding_list).save_parquet(self.config.embedding,append = os.path.exists(self.config.embedding))
        self.config.console.print('[bold green]High level elements stored[/bold green]')
            
    @info_timer(message='Summary Generation Pipeline')        
    async def main(self):
        if os.path.exists(self.config.graph_path):
            if os.path.exists(self.config.summary_path):
                os.remove(self.config.summary_path)
            await self.generate_high_level_element_summary()
            await self.high_level_element_summary()
            self.store_high_level_elements()
            self.store_graph()
            self.indices.store_all_indices(self.config.indices_path)
            self.delete_community_cache()
            
        
            
            
            
                
                    
            
            
            
    
    
            
             
        
            
            
            
        
       
        

        
