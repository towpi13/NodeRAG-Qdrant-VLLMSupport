import networkx as nx
import numpy as np
import math
import asyncio
import os
from sortedcontainers import SortedDict
from rich.console import Console
from neo4j import GraphDatabase

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
        import traceback
        try:
            average_degree = sum(dict(self.G.degree()).values())/self.G.number_of_nodes()
        except Exception as e:
            self.console.print('[bold red]Error calculating average degree:[/bold red]', e)
            self.console.print(traceback.format_exc())
            average_degree = 0
        return average_degree
    
    def defult_k(self):
        import traceback
        try:
            k = round(np.log(self.G.number_of_nodes())*self.avarege_degree()**(1/2))
        except Exception as e:
            self.console.print('[bold red]Error calculating default k:[/bold red]', e)
            self.console.print(traceback.format_exc())
            k = 0
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
        
        self.graph_db_type = getattr(self.config, 'graph_db_type', 'networkx')
        self.db_session = None
        self.G = None

        if self.graph_db_type == 'neo4j':
            # This pipeline runs after the graph is built, so we connect to read from it.
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.db_session = self.driver.session()
            # The mapper is still needed to get text for nodes already saved to parquet
            self.mapper = Mapper([self.config.entities_path,self.config.relationship_path,self.config.semantic_units_path])
        else: # networkx mode
            # Load the graph from the pickle file
            self.G = storage.load(self.config.graph_path)
            self.mapper = Mapper([self.config.entities_path,self.config.relationship_path,self.config.semantic_units_path])


    def get_important_nodes(self):
        
        temp_G = nx.Graph()
        if self.graph_db_type == 'neo4j':
            # Fetch all nodes and relationships to build a temporary graph for analysis
            records = self.db_session.run("""
                MATCH (n)-[r]->(m)
                RETURN n.hash_id AS source_id, labels(n)[0] as source_type, n.weight AS source_weight,
                       m.hash_id AS target_id, labels(m)[0] as target_type, m.weight AS target_weight
            """)
            for record in records:
                temp_G.add_node(record['source_id'], type=record['source_type'].lower(), weight=record['source_weight'])
                temp_G.add_node(record['target_id'], type=record['target_type'].lower(), weight=record['target_weight'])
                temp_G.add_edge(record['source_id'], record['target_id'])
        else: # networkx
            temp_G = self.G

        node_importance = NodeImportance(temp_G, self.config.console)
        important_nodes = node_importance.main()
        
        # This part remains the same, checking against already processed attributes
        if os.path.exists(self.config.attributes_path):
            attributes = storage.load(self.config.attributes_path)
            existing_nodes = attributes['node'].tolist()
            important_nodes = [node for node in important_nodes if node not in existing_nodes]
        
        self.important_nodes = important_nodes
        self.console.print('[bold green]Important nodes found[/bold green]')
    
    def get_neighbours_material(self,node:str):
       
        # This part is fine, as the 'important node' itself will be in the mapper
        entity = self.mapper.get(node,'context')
        
        semantic_neighbours = ''+'\n'
        relationship_neighbours = ''+'\n'
       
        if self.graph_db_type == 'neo4j':
            # Query Neo4j not just for the neighbor's ID, but also for its type and context.
            # This makes the function independent of the mapper for neighbor data.
            query = """
            MATCH (n {hash_id: $id})--(neighbor)
            RETURN labels(neighbor)[0] AS neighbor_type, neighbor.context AS neighbor_context
            """
            records = self.db_session.run(query, id=node)
            
            for record in records:
                neighbor_type = record['neighbor_type']
                neighbor_context = record['neighbor_context']
                
                # We need to map the Label from Neo4j to your internal type names
                type_map = {
                    "Entity": "entity",
                    "SemanticUnit": "semantic_unit",
                    "Relationship": "relationship",
                }
                mapped_type = type_map.get(neighbor_type)

                if neighbor_context: # Only add if context exists
                    if mapped_type == 'semantic_unit':
                        semantic_neighbours += f'{neighbor_context}\n'
                    elif mapped_type == 'relationship':
                        relationship_neighbours += f'{neighbor_context}\n'

        else: # The original networkx logic is still needed as a fallback
            for neighbour in self.G.neighbors(node):
                # Using .get() on the dictionary is safer than direct access
                neighbour_data = self.mapper.get(neighbour)
                if neighbour_data:
                    if neighbour_data.get('type') == 'semantic_unit':
                        semantic_neighbours += f'{neighbour_data.get("context", "")}\n'
                    elif neighbour_data.get('type') == 'relationship':
                        relationship_neighbours += f'{neighbour_data.get("context", "")}\n'
       
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
            query = self.get_important_neibours_material(node) # This also needs refactoring, but we'll assume it works for now
            
        response = await self.API_client({'query':query})
        if response is not None:
            attribute = Attribute(response,node)
            self.attributes.append(attribute)
            
            if self.graph_db_type == 'neo4j':
                self.db_session.run("""
                    MATCH (e:Entity {hash_id: $node_id})
                    MERGE (a:Attribute {hash_id: $attr_hash_id})
                    ON CREATE SET a.context = $attr_context, a.type = 'attribute', a.weight = 1
                    MERGE (e)-[:HAS_ATTRIBUTE]->(a)
                    SET e.attributes = CASE WHEN e.attributes IS NULL THEN [$attr_hash_id] ELSE e.attributes + $attr_hash_id END
                """, node_id=node, attr_hash_id=attribute.hash_id, attr_context=attribute.raw_context)
            else: # networkx
                self.G.nodes[node]['attributes'] = [attribute.hash_id]
                self.G.add_node(attribute.hash_id,type='attribute',weight=1)
                self.G.add_edge(node,attribute.hash_id,weight=1)
            
        self.config.tracker.update()


    def save_attributes(self):
        
        attributes = []
        
        for attribute in self.attributes:
            # The weight lookup needs to be conditional
            weight = 1 # Default weight
            if self.graph_db_type == 'networkx':
                 weight = self.G.nodes[attribute.node]['weight']

            attributes.append({'node':attribute.node,
                               'type':'attribute',
                                 'context':attribute.raw_context,
                                 'hash_id':attribute.hash_id,
                                 'human_readable_id':attribute.human_readable_id,
                                 'weight': weight,
                                 'embedding':None})
        
        # This part is now unconditional
        if not attributes:
            self.console.print("[yellow]No new attributes were generated, but creating an empty cache file for consistency.[/yellow]")
        
        storage(attributes).save_parquet(self.config.attributes_path,append= os.path.exists(self.config.attributes_path))
        self.console.print('[bold green]Attributes stored[/bold green]')
        
        
    def save_graph(self):
        if self.graph_db_type == 'networkx':
            storage(self.G).save_pickle(self.config.graph_path)
            self.config.console.print('Graph stored')
        else:
            # In neo4j mode, updates are live. We just close the connection.
            if self.driver:
                self.driver.close()
            self.config.console.print('Neo4j connection closed by Attribute pipeline.')
        
    @info_timer(message='Attribute Generation')
    async def main(self):
        
        # The check for the graph path is removed. The pipeline now runs in both modes.
        self.get_important_nodes()

        # The rest of the logic can run if there are important nodes
        if self.important_nodes:
            await self.generate_attribution_main()
            self.save_graph()
        
        # We call save_attributes unconditionally outside the if block
        # to ensure the file is always created.
        self.save_attributes() 
        self.indices.store_all_indices(self.config.indices_path)
            

        
                               
        
        
            
                
                
        
        