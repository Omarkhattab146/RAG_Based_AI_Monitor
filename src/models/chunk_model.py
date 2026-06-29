# Bulk insert/delete chunks, linked to projects via
from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk
from bson.objectid import ObjectId
from pymongo import InsertOne
import logging

logger = logging.getLogger("uvicorn.error")

class chunkModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client["data_chunks"] # select the "data_chunks" collection in the database
    
    @classmethod  # to compine __int__ which isn't async with async init_collection
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if "data_chunks" not in all_collections:
            await self.db_client.create_collection("data_chunks")
            indexes = DataChunk.get_indexs()
            for index in indexes:
                await self.collection.create_index(index["key"], name=index["name"], unique=index["unique"])

        
    async def create_chunk(self, chunk:DataChunk):
        """Reserved for future features: Individual chunk creation."""
        resualt = await self.collection.insert_one(chunk.model_dump(by_alias=True, exclude_unset=True))
        chunk._id = resualt.inserted_id  # set the chunk ID
        return chunk
    
    async def get_chunks_by_project_id(self, chunk_id: str):
        """Reserved for future features: Chunk retrieval by ID."""
        result = await self.collection.find({"_id": ObjectId(chunk_id)})  # find chunks by project ID, find as in mongoDB, await because it's async
        
        if result is None:
            return None
        
        return DataChunk(**result)
    
    async def insert_many_chunks (self, chunks: list, batch_size: int = 3):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            operations = [InsertOne(chunk.model_dump(by_alias=True, exclude_unset=True)) for chunk in batch]
            await self.collection.bulk_write(operations)  # bulk_write as in mongoDB, await because it's async

        return len(chunks)
    
    async def get_project_chunks(self, project_id: ObjectId, page_no: int=1, page_size: int=50):
        records = await self.collection.find({
                    "chunk_project_id": project_id
                }).sort([
                    ("chunk_asset_id", 1),
                    ("chunk_order", 1),
                ]).skip(
                    (page_no-1) * page_size
                ).limit(page_size).to_list(length=None)
        return [
            DataChunk(**record)
            for record in records
        ]

               