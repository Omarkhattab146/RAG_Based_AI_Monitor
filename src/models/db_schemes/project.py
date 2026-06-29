# scheme for project collection in mongoDB
from pydantic import BaseModel , Field, validator
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel): # scheme for collection
    # MongoDB document ID, this means that when we get data from mongoDB the _id field will be mapped to id attribute in pydantic model
    id: Optional[ObjectId] = Field(None,alias="_id")  
    # Human-readable project ID (must be alphanumeric)
    project_id: str = Field(..., min_length=1)        
    
    class Config:
        # allow non-pydantic types like ObjectId
        arbitrary_types_allowed = True
        # Allow population by field name (for alias mapping)
        populate_by_name = True  

    @classmethod
    def get_indexs(cls):
        return [{
            "key": [
              ("project_id", 1)  # Create ascending index on project_id field (1 = ascending, -1 = descending)
                ],
            "name": "project_id_index_1", # Human-readable index name in MongoDB
            "unique": True  # ensures no two documents can have the same
        }]
    # Real Example:
'''
1 million projects without index = scan 500K documents on average ❌
1 million projects with index = ~20 lookups ✅

Fast Lookups: Without an index, MongoDB must scan through all documents sequentially (collection scan). With 1M projects, that's ~500K document scans on average. With an index, it performs ~20 lookups using a B-tree structure. This is 50,000x faster.
'''

