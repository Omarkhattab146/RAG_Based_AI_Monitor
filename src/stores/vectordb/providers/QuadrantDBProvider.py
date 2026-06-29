from ..vectorDBinterface import VectorDBInterface
import logging
from ..vectorDBenums import DistanceMethonEnum
from qdrant_client import models , QdrantClient
from typing import List, Dict
from models.db_schemes import RetrievedDocument
class QuadrantDBProvider(VectorDBInterface):
    
    def __init__(self,db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distnace_method = None

        if distance_method == DistanceMethonEnum.COSINE.value:
            self.distnace_method = models.Distance.COSINE

        elif distance_method == DistanceMethonEnum.DOT.value:
            self.distnace_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None
    
    def is_collection_exist(self, collection_name: str):
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self)-> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name) -> dict:
        return self.client.get_collection(collection_name= collection_name)
    
    def delete_collections(self, collection_name):
        if self.is_collection_exist(collection_name=collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self,collection_name:str,
                          embeddiong_size:int,do_reset:bool = False):
        if do_reset == True:
            _ = self.delete_collections(collection_name=collection_name)
        
        if not self.is_collection_exist(collection_name=collection_name):
            _ = self.client.create_collection(
                collection_name = collection_name,
                vectors_config = models.VectorParams(
                    size = embeddiong_size,
                    distance = self.distnace_method
                )
            )
            return True
        return False
    
    def insert_one(self, collection_name:str, text:str , vector:list,
                    metadata: dict = None,
                    record_id: str = None):
        if not self.is_collection_exist(collection_name=collection_name):
            self.logger.error("This collection isn't existed")
            return False
        
        _ = self.client.upsert(
            collection_name= collection_name,
            points=[
                models.Record(
                    vector= vector,
                    payload= {
                        "text": text, "metadata": metadata 
                    }
                )
            ]
            )
        return True
    
    def insert_many(self, collection_name:str, texts:list , vectors:list,
                    metadata: list = None,
                    record_ids: list = None,
                    batch_size: int = 3):
        if not self.is_collection_exist(collection_name=collection_name):
            self.logger.error("This collection isn't existed")
            return False
        if metadata is None:
            metadata = [None] * len(texts)
        
        if record_ids == None:
            record_ids = list(range(0,len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vec = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector= batch_vec[x],
                    payload= {
                        "text": batch_texts[x], "metadata": batch_metadata[x] 
                    }
                )
                for x in range(len(batch_texts))
            ]
            try:
                _ = self.client.upsert(
                   collection_name= collection_name,
                   points= batch_records
                   )
            except Exception as e:
                self.logger.error(f"error in insert many — batch {i}:{batch_end} | reason: {e}")
                return False
        
        return True
    
    def search_by_vector(self, collection_name:str, vector:list, limit:int=10):
        results = self.client.query_points(
            collection_name=collection_name ,
            query= vector,
            limit= limit
        ).points

        if not results or len(results) == 0:
            return False
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            })
            for result in results
        ]


    
    
    
    


        
    
