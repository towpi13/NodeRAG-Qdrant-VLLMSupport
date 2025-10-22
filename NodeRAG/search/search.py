import os
from typing import Dict,List,Tuple
import numpy as np
import re
from neo4j import GraphDatabase
from qdrant_client import AsyncQdrantClient
import asyncio
import networkx as nx
import uuid

from ..storage import Mapper
from ..utils import HNSW
from ..storage import storage
from ..utils.graph_operator import GraphConcat
from ..config import NodeConfig
from ..utils.PPR import sparse_PPR
from .Answer_base import Answer,Retrieval


class NodeSearch():
    def __init__(self,config:NodeConfig):
        
        self.config = config
        
        # Graph Database Setup (no changes here)
        self.graph_db_type = getattr(self.config, 'graph_db_type', 'networkx')
        self.driver = None
        self.db_session = None
        if self.graph_db_type == 'neo4j':
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.db_session = self.driver.session()
            self.G = None
        else:
            self.G = self.load_graph()
            self.sparse_PPR = sparse_PPR(self.G)

        # Vector Store Setup
        self.vector_store = getattr(self.config, 'vector_store', 'hnsw')
        self.hnsw = None
        self.qdrant_client = None 
        
        # Mapper and text map logic remains the same
        self.mapper = self.load_mapper()
        if self.mapper:
            print("✅ Local cache (Mapper) loaded successfully.")
            self.id_to_text, self.accurate_id_to_text = self.mapper.generate_id_to_text(['entity','high_level_element_title'])
        else:
            print("⚠️ Could not load local cache. Will fall back to database queries for text content.")
            self.id_to_text, self.accurate_id_to_text = {}, {}
            
        self._semantic_units = None

    async def _get_texts_from_db_async(self, ids: List[str], qdrant_client: AsyncQdrantClient):
        """
        Modified to accept the active Qdrant client for this specific request.
        """
        if not ids:
            return {}

        texts = {}
        
        if self.graph_db_type == 'neo4j' and self.db_session:
            neo4j_query = "UNWIND $ids AS nodeId MATCH (n {hash_id: nodeId}) WHERE n.context IS NOT NULL RETURN n.hash_id AS hash_id, n.context AS context"
            results = self.db_session.run(neo4j_query, ids=ids)
            for record in results:
                texts[record['hash_id']] = record['context']

        # Qdrant part now uses the passed-in client
        if self.vector_store == 'qdrant' and qdrant_client:
            qdrant_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str)) for id_str in ids]
            
            retrieved_points = await qdrant_client.retrieve(
                collection_name=self.config.qdrant_collection_name,
                ids=qdrant_ids,
                with_payload=True
            )
            for point in retrieved_points:
                if point.payload and 'context' in point.payload and 'hash_id' in point.payload:
                    texts[point.payload['hash_id']] = point.payload['context']

        return texts 
        
    async def _async_vector_search(self, query_embedding: np.ndarray) -> list:
        """
        An asynchronous helper method to perform the vector search against Qdrant.
        """
        search_results = await self.qdrant_client.search(
            collection_name=self.config.qdrant_collection_name,
            query_vector=query_embedding,
            limit=self.config.HNSW_results
        )
        # Parse the Qdrant response to match the expected format (score, id)
        return [(point.score, point.payload['hash_id']) for point in search_results]
    

    def load_mapper(self) -> Mapper | None:
        # THIS IS THE CRUCIAL UPDATE
        # This function now returns None on failure instead of crashing.
        try:
            mapping_list = [self.config.semantic_units_path,
                            self.config.entities_path,
                            self.config.relationship_path,
                            self.config.attributes_path,
                            self.config.high_level_elements_path,
                            self.config.text_path,
                            self.config.high_level_elements_titles_path]
            
            for path in mapping_list:
                if not os.path.exists(path):
                    # Instead of raising an exception, we log it and return None
                    print(f"Cache file not found: {path}. Mapper will not be loaded.")
                    return None
            
            return Mapper(mapping_list)
        except Exception as e:
            print(f"An unexpected error occurred while loading the Mapper: {e}")
            return None
    
    def load_hnsw(self) -> HNSW:
        if os.path.exists(self.config.HNSW_path):
            hnsw = HNSW(self.config)
            hnsw.load_HNSW()
            return hnsw
        else:
            raise Exception('No HNSW data found.')
        
    def load_graph(self):
        # This function is now only called for the NetworkX backend.
        if self.graph_db_type == 'networkx':
            if os.path.exists(self.config.base_graph_path):
                G = storage.load(self.config.base_graph_path)
            else:
                raise Exception('No base graph found.')
            
            if os.path.exists(self.config.hnsw_graph_path):
                HNSW_graph = storage.load(self.config.hnsw_graph_path)
            else:
                raise Exception('No HNSW graph found.')
            
            if self.config.unbalance_adjust:
                    G = GraphConcat(G).concat(HNSW_graph)
                    return GraphConcat.unbalance_adjust(G)
                
            return GraphConcat(G).concat(HNSW_graph)
        
        # In Neo4j mode, this function is not used to load the main graph.
        return None


    async def _async_search(self, query: str):
        """
        The top-level async helper. It now correctly manages the lifecycle of the async client
        for each individual request to prevent "Event loop is closed" errors.
        """
        # Initialize a new client for this specific request.
        client = None
        if self.vector_store == 'qdrant':
            client = AsyncQdrantClient(
                url=self.config.qdrant_url, 
                api_key=self.config.qdrant_api_key,
                timeout=30.0
            )
        
        try:
            # Step 1: Initialize Retrieval object
            retrieval = Retrieval(self.config)
            retrieval.accurate_id_to_text = self.accurate_id_to_text

            # Step 2: Perform vector search
            embedding_list = self.config.embedding_client.request(query)
            query_embedding = np.array(embedding_list[0], dtype=np.float32)

            HNSW_results = []
            if self.vector_store == 'qdrant':
                # Use the new, request-specific client
                search_results = await client.search(
                    collection_name=self.config.qdrant_collection_name,
                    query_vector=query_embedding,
                    limit=self.config.HNSW_results
                )
                HNSW_results = [(point.score, point.payload['hash_id']) for point in search_results]
            else:
                HNSW_results = self.hnsw.search(query_embedding, HNSW_results=self.config.HNSW_results)
            
            retrieval.HNSW_results_with_distance = HNSW_results
            
            # Steps 3, 4, 5, 6... (The rest of the synchronous logic is the same)
            decomposed_entities = self.decompose_query(query)
            accurate_results = self.accurate_search(decomposed_entities)
            retrieval.accurate_results = accurate_results
            personlization = {ids: self.config.similarity_weight for ids in retrieval.HNSW_results}
            personlization.update({id: self.config.accuracy_weight for id in retrieval.accurate_results})
            weighted_nodes = self.graph_search(personlization)
            retrieval = self.post_process_top_k(weighted_nodes, retrieval)
            
            # Step 7: Populate final data maps
            final_node_ids = list(retrieval.unique_search_list)
            
            # 7a. Populate id_to_type map (no change)
            if self.graph_db_type == 'networkx':
                retrieval.id_to_type = { node_id: self.G.nodes[node_id].get('type') for node_id in final_node_ids if node_id in self.G }
            elif self.graph_db_type == 'neo4j' and final_node_ids:
                type_query = "UNWIND $ids AS nodeId MATCH (n {hash_id: nodeId}) RETURN n.hash_id AS hash_id, labels(n)[0] AS type"
                results = self.db_session.run(type_query, ids=final_node_ids)
                type_map = {"Entity": "entity", "SemanticUnit": "semantic_unit", "Relationship": "relationship", "HighLevelElementTitle": "high_level_element_title", "Attribute": "attribute"}
                retrieval.id_to_type = {record['hash_id']: type_map.get(record['type'], record['type'].lower()) for record in results}

            # 7b. Populate id_to_text map
            if self.graph_db_type == 'networkx':
                retrieval.id_to_text = { node_id: self.mapper.get(node_id, 'context') for node_id in final_node_ids if self.mapper.get(node_id, 'context') }
            elif self.graph_db_type == 'neo4j':
                # Pass the new, request-specific client to the fallback function
                retrieval.id_to_text = await self._get_texts_from_db_async(final_node_ids, qdrant_client=client)

            return retrieval

        finally:
            # Ensure the client connection is always closed at the end of the request
            if client:
                await client.close()


    def search(self, query: str):
        """
        The main synchronous search function.
        It now acts as a simple wrapper that calls asyncio.run() only ONCE.
        """
        return asyncio.run(self._async_search(query))
    

    def decompose_query(self,query:str):
        
        prompt = self.config.prompt_manager.decompose_query.format(query=query)
        response = self.config.API_client.request({'query':prompt,'response_format':self.config.prompt_manager.decomposed_text_json})
        
        # Case 1: The response is the expected dictionary format -> {'elements': [...]}
        if isinstance(response, dict):
            # Use .get() for safe access in case the 'elements' key is missing
            return response.get('elements', [])

        # Case 2: The response is a list containing the dictionary -> [{'elements': [...]}]
        # This checks if it's a non-empty list and the first item is a dictionary
        if isinstance(response, list) and response and isinstance(response[0], dict):
            return response[0].get('elements', [])

        # Case 3: The response is just the list of elements directly -> [...]
        # This is the most likely cause of your error.
        if isinstance(response, list):
            return response

        # Fallback: If the response is in an unknown format, log a warning and return an empty list
        # to prevent the program from crashing.
        print(f"WARN: Unexpected format from API in decompose_query. Response: {response}")
        return []    
    
    def accurate_search(self, entities: List[str]) -> List[str]:
        accurate_results = []
        
        for entity in entities:
            # Split entity into words and create a pattern to match the whole phrase
            words = entity.lower().split()
            pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, words)) + r'\b')
            result = [id for id, text in self.accurate_id_to_text.items() if pattern.search(text.lower())]
            if result:
                accurate_results.extend(result)
        
        return accurate_results
    
    
    def answer(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else:
            retrieved_info = ans.unstructured_prompt
        
        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        ans.response = self.config.API_client.request({'query':query})
        
        return ans
    
    
    
    async def answer_async(self,query:str,id_type:bool=True):
        
        
        retrieval = self.search(query)
        
        ans = Answer(query,retrieval)
        
        if id_type:
            retrieved_info = ans.structured_prompt
        else    :
            retrieved_info = ans.unstructured_prompt

        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        
        ans.response = await self.config.API_client({'query':query})
        
        return ans
        
    
    def stream_answer(self,query:str,retrieved_info:str):
        
        query = self.config.prompt_manager.answer.format(info=retrieved_info,query=query)
        response = self.config.API_client.stream_chat({'query':query})
        yield from response


    def graph_search(self,personlization:Dict[str,float])->List[Tuple[str,str]]|List[str]:
        
        if self.graph_db_type == 'networkx':
            page_rank_scores = self.sparse_PPR.PPR(personlization,alpha=self.config.ppr_alpha,max_iter=self.config.ppr_max_iter)
            return [id for id,score in page_rank_scores]
        
        elif self.graph_db_type == 'neo4j':
            personalized_ids = list(personlization.keys())
            
            # Using a simpler, non-deprecated OPTIONAL MATCH query
            query = """
                MATCH (n) WHERE n.hash_id IN $ids
                OPTIONAL MATCH (n)-[r]-(neighbor)
                RETURN n.hash_id AS source_id, neighbor.hash_id AS target_id
            """
            results = self.db_session.run(query, ids=personalized_ids)
            
            temp_G = nx.Graph()
            for record in results:
                source_id = record["source_id"]
                target_id = record["target_id"]
                if source_id:
                    temp_G.add_node(source_id)
                if target_id:
                    temp_G.add_node(target_id)
                if source_id and target_id:
                    temp_G.add_edge(source_id, target_id)
            
            if not temp_G.nodes:
                return []

            filtered_personlization = {
                node_id: prob 
                for node_id, prob in personlization.items() 
                if node_id in temp_G
            }

            if not filtered_personlization:
                return list(temp_G.nodes())

            temp_ppr = sparse_PPR(temp_G)
            page_rank_scores = temp_ppr.PPR(filtered_personlization, alpha=self.config.ppr_alpha, max_iter=self.config.ppr_max_iter)
            
            return [id for id, score in page_rank_scores]
            
    
    def post_process_top_k(self,weighted_nodes:List[str],retrieval:Retrieval)->Retrieval:
            
            # =================================================================
            # ==                  LOGIC FOR NETWORKX BACKEND                 ==
            # =================================================================
            if self.graph_db_type == 'networkx':
                
                entity_list = []
                high_level_element_title_list = []
                relationship_list = []
            
                addition_node = 0
                
                for node in weighted_nodes:
                    if node not in retrieval.search_list:
                        # Access node properties directly from the in-memory graph object
                        type = self.G.nodes[node].get('type')
                        match type:
                            case 'entity':
                                if node not in entity_list and len(entity_list) < self.config.Enode:
                                    entity_list.append(node)
                            case 'relationship':
                                if node not in relationship_list and len(relationship_list) < self.config.Rnode:
                                    relationship_list.append(node)
                            case 'high_level_element_title':
                                if node not in high_level_element_title_list and len(high_level_element_title_list) < self.config.Hnode:
                                    high_level_element_title_list.append(node)
                
                            case _:
                                if addition_node < self.config.cross_node:
                                    if node not in retrieval.unique_search_list:
                                        retrieval.search_list.append(node)
                                        retrieval.unique_search_list.add(node)
                                        addition_node += 1
                        
                        if (addition_node >= self.config.cross_node 
                            and len(entity_list) >= self.config.Enode  
                            and len(relationship_list) >= self.config.Rnode 
                            and len(high_level_element_title_list) >= self.config.Hnode):
                            break
                
                for entity in entity_list:
                    # Get related attributes from the graph object
                    attributes = self.G.nodes[entity].get('attributes')
                    if attributes:
                        for attribute in attributes:
                            if attribute not in retrieval.unique_search_list:
                                retrieval.search_list.append(attribute)
                                retrieval.unique_search_list.add(attribute)

                for high_level_element_title in high_level_element_title_list:
                    # Get related nodes from the graph object
                    related_node = self.G.nodes[high_level_element_title].get('related_node')
                    if related_node not in retrieval.unique_search_list:
                        retrieval.search_list.append(related_node)
                        retrieval.unique_search_list.add(related_node)
                    
                retrieval.relationship_list = list(set(relationship_list))
                
                return retrieval

            # =================================================================
            # ==                    LOGIC FOR NEO4J BACKEND                  ==
            # =================================================================
            elif self.graph_db_type == 'neo4j':
                
                if not weighted_nodes:
                    return retrieval
                
                # 1. Fetch properties for all top-ranked nodes in a single efficient query
                query = """
                UNWIND $ids AS nodeId
                MATCH (n {hash_id: nodeId})
                RETURN n.hash_id AS hash_id, 
                    labels(n)[0] AS type, 
                    n.attributes AS attributes, 
                    n.related_node AS related_node
                """
                # Create a dictionary to preserve the order from weighted_nodes
                results = self.db_session.run(query, ids=weighted_nodes)
                node_properties = {record['hash_id']: record for record in results}

                entity_list = []
                high_level_element_title_list = []
                relationship_list = []
                addition_node = 0

                # 2. Iterate through the nodes in their ranked order
                for node_id in weighted_nodes:
                    if node_id not in retrieval.search_list and node_id in node_properties:
                        props = node_properties[node_id]
                        node_type_from_db = props.get('type')

                        # Map Neo4j Label (e.g., 'Entity') to your internal type name (e.g., 'entity')
                        type_map = {
                            "Entity": "entity",
                            "SemanticUnit": "semantic_unit", # Assuming this label exists
                            "Relationship": "relationship",
                            "HighLevelElementTitle": "high_level_element_title"
                        }
                        mapped_type = type_map.get(node_type_from_db)
                        
                        match mapped_type:
                            case 'entity':
                                if node_id not in entity_list and len(entity_list) < self.config.Enode:
                                    entity_list.append(node_id)
                            case 'relationship':
                                if node_id not in relationship_list and len(relationship_list) < self.config.Rnode:
                                    relationship_list.append(node_id)
                            case 'high_level_element_title':
                                if node_id not in high_level_element_title_list and len(high_level_element_title_list) < self.config.Hnode:
                                    high_level_element_title_list.append(node_id)
                            case _:
                                if addition_node < self.config.cross_node:
                                    if node_id not in retrieval.unique_search_list:
                                        retrieval.search_list.append(node_id)
                                        retrieval.unique_search_list.add(node_id)
                                        addition_node += 1
                        
                        if (addition_node >= self.config.cross_node 
                            and len(entity_list) >= self.config.Enode  
                            and len(relationship_list) >= self.config.Rnode 
                            and len(high_level_element_title_list) >= self.config.Hnode):
                            break
                
                # 3. Process the collected lists to find their related nodes
                for entity_id in entity_list:
                    # Get related attributes from the pre-fetched properties
                    if entity_id in node_properties and node_properties[entity_id].get('attributes'):
                        for attribute in node_properties[entity_id]['attributes']:
                            if attribute not in retrieval.unique_search_list:
                                retrieval.search_list.append(attribute)
                                retrieval.unique_search_list.add(attribute)
                
                for title_id in high_level_element_title_list:
                    # Get related nodes from the pre-fetched properties
                    if title_id in node_properties and node_properties[title_id].get('related_node'):
                        related_node = node_properties[title_id]['related_node']
                        if related_node not in retrieval.unique_search_list:
                            retrieval.search_list.append(related_node)
                            retrieval.unique_search_list.add(related_node)

                retrieval.relationship_list = list(set(relationship_list))
                
                return retrieval
        
    