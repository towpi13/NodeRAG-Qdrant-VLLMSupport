import os
import json
from ...config import NodeConfig
from ...logging import info_timer
from ...storage import genid




class INIT_pipeline():
    def __init__(self, config:NodeConfig):
        
        self.config = config
        self.documents_path = []
        
    @property
    def document_path_hash(self):
        if self.documents_path is None:
            raise ValueError('Document path is not loaded')
        else:
            return genid(''.join(self.documents_path),"sha256")
    
    def check_folder_structure(self):
        if not os.path.exists(self.config.main_folder):
            raise ValueError(f'Main folder {self.config.main_folder} does not exist')
        
        if not os.path.exists(self.config.input_folder):
            raise ValueError(f'Input folder {self.config.input_folder} does not exist')
        
    def load_files(self):
        
        
        if self.config.docu_type == 'mixed':
            for file in os.listdir(self.config.input_folder):
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(self.config.input_folder, file)
                    self.documents_path.append(file_path)
        else:
            for file in os.listdir(self.config.input_folder):
                if file.endswith(f'.{self.config.docu_type}'):
                    file_path = os.path.join(self.config.input_folder, file)
                    self.documents_path.append(file_path)
                    
        if len(self.documents_path) == 0:
            raise ValueError(f'No files found in {self.config.input_folder}')
        
    def check_increment(self):
        if not os.path.exists(self.config.document_hash_path):
            self.save_document_hash()
            return False
        else:
            with open(self.config.document_hash_path,'r') as f:
                file = json.load(f)
            previous_hash = file['document_path_hash']
            if previous_hash == self.document_path_hash:
                return False
            else:
                return True
            
    def save_document_hash(self):
        with open(self.config.document_hash_path,'w') as f:
            json.dump({'document_path_hash':self.document_path_hash,'document_path':self.documents_path},f)
     
   
    @info_timer(message='Init Pipeline')
    async def main(self):
        self.check_folder_structure()
        self.load_files()
        if self.check_increment():
            self.save_document_hash()
            return True
        else:
            return False
        
        
        
