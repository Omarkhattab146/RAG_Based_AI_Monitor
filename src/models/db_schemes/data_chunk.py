# scheme for collection chunks in mongoDB
from pydantic import BaseModel , Field, validator
from typing import Optional
from bson.objectid import ObjectId

# Purpose: Structure for text chunks created from splitting documents
class DataChunk(BaseModel): # scheme for collection chunks
    id: Optional[ObjectId] = Field(None,alias="_id")  # MongoDB document ID, this means that when we get data from mongoDB the _id field will be mapped to id attribute in pydantic model
    
    chunk_text: str = Field(..., min_length=1)
    
    chunk_metadata: dict
    
    chunk_order: int=Field(..., ge=0)  # greater than or equal to 0

    chunk_project_id: ObjectId # to link chunk to project
    
    chunk_asset_id: ObjectId # to link chunk to asset/file


    class Config:
        arbitrary_types_allowed = True  

    
    @classmethod
    def get_indexs(cls):
        return [{
            "key": [
              ("chunk_project_id", 1)
                ],
            "name": "chunk_project_id_index_1",
            "unique": False
        }]
        
class RetrievedDocument(BaseModel):
    text: str
    score: float
