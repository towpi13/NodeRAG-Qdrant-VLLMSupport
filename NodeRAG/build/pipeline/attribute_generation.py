import networkx as nx
import numpy as np
import math
import asyncio
import os
from sortedcontainers import SortedDict
from rich.console import Console


from ...storage import (
    Mapper,
    storage
)
from ..component import Attribute
from ...config import NodeConfig
from ...logging import info_timer



class NodeImportance:
    
    def __init__(self,graph:nx.Graph,console:Console):
        self.G = graph
        self.important_nodes = []
        self.console = console
        
    def K_core(self,k:int|None = None):
        
        if k is None:
            k = self.defult_k()
        
        self.k_subgraph = nx.core.k_core(self.G,k=k)
        
        for nodes in self.k_subgraph.nodes():
            if self.G[nodes]['type'] == 'entity' and self.G[nodes]['weight'] > 1:
                self.important_nodes.append(nodes)
        
    def avarege_degree(self):
        average_degree = sum(dict(self.G.degree()).values())/self.G.number_of_nodes()
        return average_degree
    
    def defult_k(self):
        k = round(np.log(self.G.number_of_nodes())*self.avarege_degree()**(1/2))
        return k
    
    def betweenness_centrality(self):
        
        self.betweenness = nx.betweenness_centrality(self.G,k=10)
        average_betweenness = sum(self.betweenness.values())/len(self.betweenness)
        scale = round(math.log10(len(self.betweenness)))
        
        for node in self.betweenness:
            if self.betweenness[node] > average_betweenness*scale:
                if self.G.nodes[node]['type'] == 'entity' and self.G.nodes[node]['weight'] > 1:
                    self.important_nodes.append(node)
                    
    def main(self):
        self.K_core()
        self.console.print('[bold green]K_core done[/bold green]')
        self.betweenness_centrality()
        self.console.print('[bold green]Betweenness done[/bold green]')
        self.important_nodes = list(set(self.important_nodes))
        return self.important_nodes
        
        
        
class Attribution_generation_pipeline:
            
    def __init__(self,config:NodeConfig):
        

        self.config = config
        self.prompt_manager = config.prompt_manager
        self.indices = config.indices
        self.console = config.console
        self.API_client = config.API_client
        self.token_counter = config.token_counter
        self.important_nodes = []
        self.attributes = []
        
        
        self.mapper = Mapper([self.config.entities_path,self.config.relationship_path,self.config.semantic_units_path])
        self.G = storage.load(self.config.graph_path)
        
    def get_important_nodes(self):
        
        node_importance = NodeImportance(self.G,self.config.console)
        important_nodes = node_importance.main()
        
        if os.path.exists(self.config.attributes_path):
            attributes = storage.load(self.config.attributes_path)
            existing_nodes = attributes['node'].tolist()
            important_nodes = [node for node in important_nodes if node not in existing_nodes]
        
        self.important_nodes = important_nodes
        self.console.print('[bold green]Important nodes found[/bold green]')
    
    def get_neighbours_material(self,node:str):
       
        entity = self.mapper.get(node,'context')
        semantic_neighbours = ''+'\n'
        relationship_neighbours = ''+'\n'
       
        for neighbour in self.G.neighbors(node):
            if self.G.nodes[neighbour]['type'] == 'semantic_unit':
                semantic_neighbours += f'{self.mapper.get(neighbour,"context")}\n'
            elif self.G.nodes[neighbour]['type'] == 'relationship':
                relationship_neighbours += f'{self.mapper.get(neighbour,"context")}\n'
       
        query = self.prompt_manager.attribute_generation.format(entity = entity,semantic_units = semantic_neighbours,relationships = relationship_neighbours)
        return query
    
    
    def get_important_neibours_material(self,node:str):
        
        entity = self.mapper.get(node,'context')
        semantic_neighbours = ''+'\n'
        relationship_neighbours = ''+'\n'
        sorted_neighbours = SortedDict()
        
        for neighbour in self.G.neighbors(node):
            value = 0
            for neighbour_neighbour in self.G.neighbors(neighbour):
                value += self.G.nodes[neighbour_neighbour]['weight']
            sorted_neighbours[neighbour] = value
        
        query = ''
        for neighbour in reversed(sorted_neighbours):
            while not self.token_counter.token_limit(query):
                query = self.prompt_manager.attribute_generation.format(entity = entity,semantic_units = semantic_neighbours,relationships = relationship_neighbours)
                if self.G.nodes[neighbour]['type'] == 'semantic_unit':
                    semantic_neighbours += f'{self.mapper.get(neighbour,"context")}\n'
                elif self.G.nodes[neighbour]['type'] == 'relationship':
                    relationship_neighbours += f'{self.mapper.get(neighbour,"context")}\n'
        
        return query
    
    async def generate_attribution_main(self):
        
        tasks = []
        self.config.tracker.set(len(self.important_nodes),desc="Generating attributes")
        
        for node in self.important_nodes:
            tasks.append(self.generate_attribution(node))
        
        await asyncio.gather(*tasks)
        
        self.config.tracker.close()
                    
            
            
            
    async def generate_attribution(self,node:str):
        query = self.get_neighbours_material(node)
        
        
        if self.token_counter.token_limit(query):
            query = self.get_important_neibours_material(node)
            
        response = await self.API_client({'query':query})
        if response is not None:
            attribute = Attribute(response,node)
            
            self.attributes.append(attribute)
            self.G.nodes[node]['attributes'] = [attribute.hash_id]
            self.G.add_node(attribute.hash_id,type='attribute',weight=1)
            self.G.add_edge(node,attribute.hash_id,weight=1)
        self.config.tracker.update()

    def save_attributes(self):
        
        attributes = []
        
        for attribute in self.attributes:
            attributes.append({'node':attribute.node,
                               'type':'attribute',
                                 'context':attribute.raw_context,
                                 'hash_id':attribute.hash_id,
                                 'human_readable_id':attribute.human_readable_id,
                                 'weight':self.G.nodes[attribute.node]['weight'],
                                 'embedding':None})
        
        storage(attributes).save_parquet(self.config.attributes_path,append= os.path.exists(self.config.attributes_path))
        self.config.console.print('[bold green]Attributes stored[/bold green]')
        
        
    def save_graph(self):
        
        storage(self.G).save_pickle(self.config.graph_path)
        self.config.console.print('Graph stored')
        
    @info_timer(message='Attribute Generation')
    async def main(self):
        
        if os.path.exists(self.config.graph_path):
            
            self.get_important_nodes()
            await self.generate_attribution_main()
            self.save_attributes()
            self.save_graph()
            self.indices.store_all_indices(self.config.indices_path)
            

        
                               
        
        
            
                
                
        
        