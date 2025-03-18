import networkx as nx
from typing import List,Dict
import json
import os
import asyncio

from ...LLM import LLMOutput

from ..component import (
    Semantic_unit,
    Entity,
    Relationship
)

from ...storage import storage
from ...config import NodeConfig
from ...logging import info_timer
class Graph_pipeline:
    

    def __init__(self,config:NodeConfig):
        
        self.config = config
        self.G = self.load_graph()
        self.indices = self.config.indices
        self.data ,self.processed_data = self.load_data()
        self.API_request = self.config.API_client
        self.prompt_manager = self.config.prompt_manager
        self.semantic_units = []
        self.entities = []
        self.relationship, self.relationship_lookup = self.load_relationship()
        self.relationship_nodes = []
        self.console = self.config.console
    
    
    
    def check_processed(self,data:Dict)->bool:
        if data.get('processed'):
            return False
        return True
        
    def load_graph(self) -> nx.Graph:
        if os.path.exists(self.config.graph_path):
            return storage.load_pickle(self.config.graph_path)
        return nx.Graph()
        
    def load_data(self)->List[LLMOutput]:
        data_list = []
        processed_data = []
        with open(self.config.text_decomposition_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if self.check_processed(data):
                    data_list.append(data)
                else:
                    processed_data.append(data)
        return data_list,processed_data
    
    def load_relationship(self)->List[Relationship]:
        
        if os.path.exists(self.config.relationship_path):
            df = storage.load(self.config.relationship_path)
            relationship = [Relationship.from_df_row(row) for row in df.itertuples()]
            relationship_lookup = {relationship.hash_id: relationship for relationship in relationship}
            return relationship,relationship_lookup
        
        return [],{}
    
    async def build_graph(self):
        
        self.config.tracker.set(len(self.data),desc="Building graph")
        tasks = []
        
        for data in self.data:
            tasks.append(self.graph_tasks(data))
        await asyncio.gather(*tasks)
        self.config.tracker.close()
        
    async def graph_tasks(self,data:Dict):
        text_hash_id = data.get('text_hash_id')
        response = data.get('response')

        if isinstance(response,dict):   
            Output = response.get('Output')
            
            for output in Output:
                semantic_unit = output.get('semantic_unit')
                entities = output.get('entities')
                relationships = output.get('relationships')
                
                semantic_unit_hash_id = self.add_semantic_unit(semantic_unit,text_hash_id)
                entities_hash_id = self.add_entities(entities,text_hash_id)
        
                entities_hash_id_re = await self.add_relationships(relationships,text_hash_id)
                entities_hash_id.extend(entities_hash_id_re)
                self.add_semantic_belongings(semantic_unit_hash_id,entities_hash_id)
            data['processed'] = True
            self.config.tracker.update()
        
        
    def save_data(self):
        with open(self.config.text_decomposition_path, 'w', encoding='utf-8') as f:
            self.processed_data.extend(self.data)
            for data in self.processed_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
    
    def add_semantic_unit(self,semantic_unit:Dict,text_hash_id:str):
        
        semantic_unit = Semantic_unit(semantic_unit,text_hash_id)
        if self.G.has_node(semantic_unit.hash_id):
            self.G.nodes[semantic_unit.hash_id]['weight'] += 1
        else:
            self.G.add_node(semantic_unit.hash_id,type ='semantic_unit',weight = 1)
            self.semantic_units.append(semantic_unit)
        return semantic_unit.hash_id
        
    def add_entities(self,entities:List[Dict],text_hash_id:str):
        
        entities_hash_id = []
        
        for entity in entities:
            
            entity = Entity(entity,text_hash_id)
            entities_hash_id.append(entity.hash_id)
            
            if self.G.has_node(entity.hash_id):
                self.G.nodes[entity.hash_id]['weight'] += 1
            
            else:
                self.G.add_node(entity.hash_id,type = 'entity',weight = 1)
                self.entities.append(entity)
        
        return entities_hash_id
    
    def add_semantic_belongings(self, semantic_unit_hash_id: str, hash_id: List[str]):
        for entity_hash_id in hash_id:
            
            
            if self.G.has_edge(semantic_unit_hash_id,entity_hash_id):
                self.G[semantic_unit_hash_id][entity_hash_id]['weight'] += 1
            else:
                self.G.add_edge(semantic_unit_hash_id,entity_hash_id,weight = 1)
            
    async def add_relationships(self,relationships:List[str],text_hash_id:str):
        
        entities_hash_id = []
        for relationship in relationships:
            
            relationship = relationship.split(',')
            relationship = [i.strip() for i in relationship]
            
            if len(relationship) != 3:
                relationship = await self.reconstruct_relationship(relationship)
            
            relationship = Relationship(relationship,text_hash_id)
            hash_id = relationship.hash_id
            if hash_id in self.relationship_lookup:
                Re = self.relationship_lookup[hash_id]
                Re.add(relationship.raw_context)
                continue
            
            
            self.relationship.append(relationship)
            self.relationship_lookup[hash_id] = relationship
            
            
            for node in [relationship.source, relationship.target, relationship]:
                if not self.G.has_node(node.hash_id):
                    self.G.add_node(node.hash_id, type='entity' if node in [relationship.source, relationship.target] else 'relationship', weight=1)
                    if node in [relationship.source, relationship.target]:
                        self.relationship_nodes.append(node)
                        entities_hash_id.append(node.hash_id)
                    

            for edge in [(relationship.source.hash_id, relationship.hash_id), (relationship.hash_id, relationship.target.hash_id)]:
                if not self.G.has_edge(*edge):
                    self.G.add_edge(*edge, weight=1)
                else:
                    self.G[edge[0]][edge[1]]['weight'] += 1
        return entities_hash_id
                
    async def reconstruct_relationship(self,relationship:List[str])->List[str]:
        
        query = self.prompt_manager.relationship_reconstraction.format(relationship=relationship)
        json_format = self.prompt_manager.relationship_reconstraction_json
        input_data = {'query':query,'response_format':json_format}
        response = await self.API_request(input_data)
        return [response.get('source'),response.get('relationship'),response.get('target')]
                
            
           
    
    def save_semantic_units(self):
        semantic_units = []
        for semantic_unit in self.semantic_units:
            semantic_units.append({'hash_id':semantic_unit.hash_id,
                                   'human_readable_id':semantic_unit.human_readable_id,
                                   'type':'semantic_unit',
                                   'context':semantic_unit.raw_context,
                                   'text_hash_id':semantic_unit.text_hash_id,
                                   'weight':self.G.nodes[semantic_unit.hash_id]['weight'],
                                   'embedding':None,
                                   'insert':None})
        G_semantic_units = [node for node in self.G.nodes if self.G.nodes[node]['type'] == 'semantic_unit']
        assert len(semantic_units) == len(G_semantic_units), f"The number of semantic units is not equal to the number of nodes in the graph. {len(semantic_units)} != {len(G_semantic_units)}"
        return semantic_units
        
    
    def save_entities(self):
        entities = []
        
        for entity in self.entities:
            entities.append({'hash_id':entity.hash_id,
                             'human_readable_id':entity.human_readable_id,
                             'type':'entity',
                             'context':entity.raw_context,
                             'text_hash_id':entity.text_hash_id,
                             'weight':self.G.nodes[entity.hash_id]['weight']})
        for node in self.relationship_nodes:
            entities.append({'hash_id':node.hash_id,
                             'human_readable_id':node.human_readable_id,
                             'type':'entity',
                             'context':node.raw_context,
                             'text_hash_id':node.text_hash_id,
                             'weight':self.G.nodes[node.hash_id]['weight']})
        G_entities = [node for node in self.G.nodes if self.G.nodes[node]['type'] == 'entity']
        assert len(entities) == len(G_entities), f"The number of entities is not equal to the number of nodes in the graph. {len(entities)} != {len(G_entities)}"
        return entities
        
        
    def save_relationships(self):
        relationships = []
        for relationship in self.relationship:
            relationships.append({'hash_id':relationship.hash_id,
                                 'human_readable_id':relationship.human_readable_id,
                                 'type':'relationship',
                                 'unique_relationship':list(relationship.unique_relationship),
                                 'context':relationship.raw_context,
                                 'text_hash_id':relationship.text_hash_id,
                                 'weight':self.G.nodes[relationship.hash_id]['weight']})
        relation_nodes = [node for node in self.G.nodes if self.G.nodes[node]['type'] == 'relationship']
        assert len(relationships) == len(relation_nodes), f"The number of relationships is not equal to the number of edges in the graph. {len(relationships)} != {len(relation_nodes)}"
        return relationships
        
        
    def save(self):
        semantic_units = self.save_semantic_units()
        entities = self.save_entities()
        relationships = self.save_relationships()
        storage(semantic_units).save_parquet(self.config.semantic_units_path,append= os.path.exists(self.config.semantic_units_path))
        storage(entities).save_parquet(self.config.entities_path,append= os.path.exists(self.config.entities_path))
        storage(relationships).save_parquet(self.config.relationship_path,append= os.path.exists(self.config.relationship_path))
        self.console.print('[green]Semantic units, entities and relationships stored[/green]')
        
    def save_graph(self):
        if self.data == []:
            return None
        storage(self.G).save_pickle(self.config.graph_path)
        self.console.print('[green]Graph stored[/green]')
    
    @info_timer(message='Graph Pipeline')
    async def main(self):
        await self.build_graph()
        self.save()
        self.save_graph()
        self.indices.store_all_indices(self.config.indices_path)
        self.save_data()
        
        
        
            
    
    
                
        