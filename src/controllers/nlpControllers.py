import logging
from typing import List

from .BaseController import BaseController
from models.db_schemes import DataChunk, Project

logger = logging.getLogger(__name__)


class nlpControllers(BaseController):
    def __init__(self, vectordb_client, embedding_client):
        super().__init__()
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.logger = logger

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()

    def index_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        chunks_ids: List[int],
        do_reset: bool = False,
    ):
        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]

        vectors = self.embedding_client.embed_texts(texts)

        self.logger.warning("INDEX_VECTOR_DB CALLED")
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embeddiong_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )
        insert_result = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            record_ids=chunks_ids,
        )
        if not insert_result:
            self.logger.error(
                f"index_vector_db: insert_many failed for collection '{collection_name}'"
            )
        return insert_result

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        collection_name = self.create_collection_name(project_id=project.project_id)

        vector = self.embedding_client.embed_text(text=text, doc_type="query")

        if not vector or len(vector) == 0:
            return False

        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
        )
        if not results:
            return False

        return results
