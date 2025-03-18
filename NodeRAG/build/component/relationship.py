from typing import List
from .unit import Unit_base
from ...storage import genid
from ...utils.readable_index import relation_index
from .entity import Entity



relation_index_counter = relation_index()

class Relationship(Unit_base):
    
    def __init__(self, relationship_tuple: List[str] = None, text_hash_id: str = None, 
                 frozen_set: frozenset = None, context: str = None,human_readable_id:int = None):
        if relationship_tuple:
            self.relationship_tuple = relationship_tuple
            self.source = Entity(relationship_tuple[0], text_hash_id)
            self.target = Entity(relationship_tuple[2], text_hash_id)
            self.unique_relationship = frozenset((self.source.hash_id,self.target.hash_id))
            self.raw_context = " ".join(self.relationship_tuple)
            self._human_readable_id = None
            
        elif frozen_set:
            self.unique_relationship = frozenset(frozen_set)
            self.raw_context = context
            self._human_readable_id = human_readable_id
        else:
            raise ValueError("Must provide either relationship_tuple or (frozen_set and context)")
        
        self.text_hash_id = text_hash_id
        self._hash_id = None
        
        
    @property
    def hash_id(self):
        if not self._hash_id:
            self._hash_id = genid(list(self.unique_relationship),"sha256")
        return self._hash_id
    
    @property
    def human_readable_id(self):
        if not self._human_readable_id:
            self._human_readable_id = relation_index_counter.increment()
        return self._human_readable_id
    
    def __eq__(self, other):
        if isinstance(other, frozenset):
            return self.unique_relationship == other
        elif isinstance(other, Relationship):
            return self.unique_relationship == other.unique_relationship
        return False
    
    def __hash__(self):
        return hash(self.unique_relationship)
    
    def add(self,relationship_tuple:List[str]):
        raw_context = " ".join(relationship_tuple)
        self.raw_context = self.raw_context + "\t" + raw_context

    def __str__(self):
        return self.raw_context

    @classmethod
    def from_df_row(cls,row):
        return cls(frozen_set=row['unique_relationship'],context=row['context'],human_readable_id=row['human_readable_id'])
