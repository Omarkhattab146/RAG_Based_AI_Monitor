from typing import Optional

from .BaseDataModel import BaseDataModel
from .db_schemes.project import Project

#  Purpose: Model for CRUD operations for projects (create, get, list)
class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        # select the "projects" collection in the database
        self.collection = self.db_client["projects"] 


    # To compine __int__ which isn't async with async init_collection
    @classmethod  
    async def create_instance(cls, db_client: object):
        # Factory pattern: allows async initialization
        instance = cls(db_client=db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        # Create collection if it doesn't exist
        all_collections = await self.db_client.list_collection_names()
        
        if "projects" not in all_collections:
            await self.db_client.create_collection("projects")
            # Create indexes for performance
            indexes = Project.get_indexs()
            for index in indexes:
                await self.collection.create_index(index["key"], name=index["name"], unique=index["unique"])

    async def create_project(self, project: Project): 
       # Insert project into MongoDB
        result = await self.collection.insert_one(
            project.model_dump(by_alias=True, exclude_unset=True)
            ) 
        project.id = result.inserted_id  # Set MongoDB-generated ID
        return project
    
    
    async def get_project_or_create(self , project_id: str):
        # Retrieve project or create if doesn't exist
        record = await self.collection.find_one(
            {"project_id": project_id}
            )  # find the project by its ID, find_one as in mongoDB, await because it's async
        
        if record is None:
            # create new project, Project scheme
            project = Project(project_id=project_id) 
            project = await self.create_project(project=project)
            return project
        else:
            # Ensure _id is mapped to id for Pydantic schema
            record["_id"] = record.get("_id")
            return Project(**record)

    async def get_project_by_project_id(self, project_id: str) -> Optional[Project]:
        record = await self.collection.find_one({"project_id": project_id})
        if record is None:
            return None
        record["_id"] = record.get("_id")
        return Project(**record)

    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        """Reserved for future features: Project listing and pagination."""

        # count total number of documents
        total_docs = await self.collection.count_documents({}) 

        # calculate number of pages
        total_pages = total_docs // page_size
        if total_docs % page_size != 0:
            total_pages += 1
        

        cursor = self.collection.find({}).skip((page - 1) * page_size).limit(page_size)
        projects = []
        async for document in cursor:
            projects.append(Project(**document))
        
        return projects, total_pages
    

        