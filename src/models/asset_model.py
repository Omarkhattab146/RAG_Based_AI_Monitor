from .BaseDataModel import BaseDataModel
from .db_schemes import Asset
from bson.objectid import ObjectId


class AssetModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client["assets"] # select the "assets" collection in the database

    @classmethod  # to compine __int__ which isn't async with async init_collection
    async def create_instance(cls, db_client: object):
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if "assets" not in all_collections:
            await self.db_client.create_collection("assets")
            indexes = Asset.get_indexs()
            for index in indexes:
                await self.collection.create_index(index["key"], name=index["name"], unique=index["unique"])

    async def create_asset(self, asset: Asset): # create a new asset in the database
        result = await self.collection.insert_one(asset.model_dump(by_alias=True, exclude_unset=True))
        asset.id = result.inserted_id  # set the asset ID to the inserted ID from MongoDB (ObjectId)

        return asset # return the newly created asset with MongoDB ID

    async def get_all_project_assets(self, asset_project_id: str, asset_type: str):
        """Reserved for future features: Asset filtering and listing."""
        record = await self.collection.find(
            {
                "asset_project_id": ObjectId(asset_project_id) if isinstance(asset_project_id, str) else asset_project_id,
                "asset_type": asset_type
            }
        ).to_list(length=None) # find assets by project ID and type, find as in mongoDB, await because it's async

        return [
            Asset(**rec)
            for rec in record
        ]
    async def get_asset_by_name(self, asset_project_id: str, asset_name: str):
        """Reserved for future features: Asset retrieval by name."""
        record = await self.collection.find_one(
            {
                "asset_project_id": ObjectId(asset_project_id) if isinstance(asset_project_id, str) else asset_project_id,
                "asset_name": asset_name
            }
        ) # find asset by project ID and name, find_one as in mongoDB, await because it's async

        if record is None:
            return None
        

        return Asset(**record)
