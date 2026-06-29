from pydantic import BaseModel , Field, validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Asset(BaseModel): # scheme for collection assets
    id: Optional[ObjectId] = Field(None,alias="_id")  # MongoDB document ID, this means that when we get data from mongoDB the _id field will be mapped to id attribute in pydantic model
    asset_project_id: ObjectId 
    asset_type: str = Field(..., min_length=1)
    asset_name: str = Field(..., min_length=1)
    asset_size: int = Field(ge=0, default=None)  # greater than or equal to 0
    asset_config: dict = Field(default=None)
    asset_puplished_at: datetime = Field(default=datetime.utcnow)


    class Config:
        arbitrary_types_allowed = True
    @classmethod
    def get_indexs(cls):
        return [{
            "key": [
              ("asset_project_id", 1)  # Create ascending index on project_id field (1 = ascending, -1 = descending)
                ],
            "name": "asset_project_id_index_1", # Human-readable index name in MongoDB
            "unique": False
        }, 
        {
            "key": [
              ("asset_project_id", 1),
              ("asset_name", 1)
                ],
            "name": "asset_project_id_name_index_1",
            "unique": True
        }
        ]