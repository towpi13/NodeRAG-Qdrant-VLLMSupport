import networkx as nx
from sortedcontainers import SortedDict
import json
import backoff
from json.decoder import JSONDecodeError

from ...storage import genid
from ...utils.readable_index import community_summary_index,high_level_element_index
from .unit import Unit_base
from ...storage.graph_mapping import Mapper

community_summary_index_counter = community_summary_index()
high_level_element_index_counter = high_level_element_index()


class Community_summary(Unit_base):
        
        def __init__(self,community_node:str|None,mapper:Mapper,graph:nx.MultiGraph,config):
            
            self.community_node = community_node
            self.client = config.API_client
            self.mapper = mapper
            self.graph = graph
            self._used_unit = None
            self.prompt = config.prompt_manager
            self.token_counter = config.token_counter
            self._hash_id = None
            self._human_readable_id = None
            
        @property
        def hash_id(self):
            if not self._hash_id:
                self._hash_id = genid(self.community_node,"sha256")
            return self._hash_id
        @property
        def human_readable_id(self):
            if not self._human_readable_id:
                self._human_readable_id = community_summary_index_counter.increment()
            return self._human_readable_id
            
        @property
        def used_unit(self):
            if self._used_unit is None:
                self._used_unit = []
                for node in self.community_node:
                    if self.graph.nodes[node]['type'] == 'semantic_unit':
                        self._used_unit.append(node)
                    elif self.graph.nodes.get(node, {}).get('type') == 'attribute':
                        self._used_unit.append(node)
                    elif self.graph.nodes[node].get('attribute') == 1:
                        for neighbour in self.graph.neighbors(node):
                            if self.graph.nodes[neighbour]['type'] == 'attribute':
                                self._used_unit.append(neighbour)
            return self._used_unit
        
        def get_normal_query(self):
            content = ''
            for node in self.used_unit:
                content += self.mapper.get(node,'context')+'\n'
            query = self.prompt.community_summary.format(content = content)
            return query
        
        def get_important_node_query(self):
            weights_dict = SortedDict()
            for name in self.used_unit:
                weight = 0
                for neighbour in self.graph.neighbors(name):
                    weight += self.graph[neighbour]['weight']
                weights_dict[name] = weight
            weights_dict = reversed(weights_dict)
            query_old = ''
            for i in range(len(weights_dict)+1):
                query = self.get_query(weights_dict.keys()[:i])
                if self.token_counter.token_limit(query):
                    return query_old
                query_old = query
                
        def get_query(self):
            query = self.get_normal_query()
            if self.token_counter.token_limit(query):
                return self.get_important_node_query()
            return query
        
        @backoff.on_exception(backoff.expo,
                              (JSONDecodeError,),
                              max_tries=3,
                              max_time=15)
        async def generate_community_summary(self):
            query = self.get_query()
            input = {'query':query,'response_format':self.prompt.high_level_element_json}
            self.response = await self.client(input)
                   
            
                
class High_level_elements(Unit_base):
    def __init__(self,context:str,title:str,config):
        self.context = context
        self.title = title
        self.embedding_client = config.embedding_client
        self._hash_id = None
        self._title_hash_id = None
        self._human_readable_id = None
        self.embedding = None

    @property
    def hash_id(self):
        if not self._hash_id:
            self._hash_id = genid([self.context],"sha256")
        return self._hash_id
    
    @property
    def title_hash_id(self):
        if not self._title_hash_id:
            self._title_hash_id = genid([self.title],"sha256")
        return self._title_hash_id
    
    @property
    def human_readable_id(self):
        if not self._human_readable_id:
            self._human_readable_id = high_level_element_index_counter.increment()
        return self._human_readable_id
    
    def store_embedding(self,embedding:list[float]):
        self.embedding = embedding
        
    def related_node(self,nodes:list[str]):
        self.related_node = nodes
            
            
            
        
        
        