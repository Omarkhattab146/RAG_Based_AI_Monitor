# Loads & processes files (TXT/PDF) using LangChain, chunks documents
from .BaseController import BaseController
from fastapi import UploadFile
from .ProjectController import ProjectController
import os

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader # to load txt,PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Purpose: Load & process files (TXT/PDF) using LangChain, chunk documents
class ProcessController(BaseController):
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extenstion(self, file_id: str):
        # To retrun extension of file to know if it text or else (.txt, .pdf, etc.)
        return os.path.splitext(file_id)[-1]   

    def get_file_loader(self, file_id: str):
        file_extension = self.get_file_extenstion(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)
    
        if not os.path.exists(file_path):
            return None
        
        if file_extension == ".txt":
            return TextLoader(file_path, encoding="utf8") #encoding to encode the name so if the name contain arabic letters it will be read correctly
        elif file_extension == ".pdf":
            return PyMuPDFLoader(file_path)
        else:
            return None

    
    def get_file_content(self, file_id: str):  
        loader = self.get_file_loader(file_id=file_id)
        if loader is None:
            return None
        
        # Returns list of Document objects with page_content & metadata
        documents = loader.load()  
        return documents 
    
    def process_file_content(self,file_content:list, file_id:str, 
                             chunk_size: int=400, overlap_size: int=50):
        # Split document into overlapping chunks
        # chunk_size=10: each chunk is ~10 tokens
        # overlap_size=50: 50 chars overlap between consecutive chunks        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len
        )
        file_content_text = [
            rec.page_content
            for rec in file_content
        ]
        file_content_metadat = [
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_text,
            metadatas=file_content_metadat
        )
        
        return chunks