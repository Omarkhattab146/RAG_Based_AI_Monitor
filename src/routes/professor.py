import logging
import os
import time

import aiofiles
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse

from controllers.DataController import DataController
from controllers.ProcessController import ProcessController
from controllers.nlpControllers import nlpControllers
from helper.config import get_settings
from models.asset_model import AssetModel
from models.chunk_model import chunkModel
from models.db_schemes import Asset, DataChunk
from models.project_model import ProjectModel

logger = logging.getLogger("uvicorn.error")

professor_router = APIRouter(
    prefix="/api/v1/professor",
    tags=["Professor"],
)


@professor_router.post("/upload_docs/{project_id}")
async def professor_upload_docs(
    request: Request,
    project_id: str,
    file: UploadFile = File(...),
    app_settings=Depends(get_settings),
):
    """
    Upload study material to a project's knowledge base.
    Processes file → chunks → MongoDB + VectorDB.
    """
    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project_or_create(project_id=project_id)

        logger.info(f"📁 Professor upload: project_id={project_id}")

        is_valid = DataController().Validate_uploaded_file(file=file)
        if not is_valid:
            return JSONResponse(status_code=400, content={"signal": "File validation failed"})

        file_path, file_id = DataController().generate_unique_file_path(
            original_filename=file.filename,
            project_id=project_id,
        )

        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                    await f.write(chunk)
        except Exception as e:
            logger.error(f"❌ File save failed: {e}")
            return JSONResponse(status_code=500, content={"signal": "File save failed", "error": str(e)})

        asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
        asset_resource = Asset(
            asset_project_id=project.id,
            asset_type="file",
            asset_name=file_id,
            asset_size=os.path.getsize(file_path),
        )
        asset_record = await asset_model.create_asset(asset=asset_resource)

        process_controller = ProcessController(project_id=project_id)
        file_content = process_controller.get_file_content(file_id=file_id)
        if file_content is None:
            return JSONResponse(status_code=404, content={"signal": f"Unsupported file format: {file_id}"})

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=400,
            overlap_size=50,
        )
        if file_chunks is None:
            return JSONResponse(status_code=500, content={"signal": "Processing failed"})

        chunk_model_instance = await chunkModel.create_instance(db_client=request.app.db_client)
        chunk_records = [
            DataChunk(
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i + 1,
                chunk_project_id=project.id,
                chunk_asset_id=asset_record.id,
            )
            for i, chunk in enumerate(file_chunks)
        ]
        chunks_inserted = await chunk_model_instance.insert_many_chunks(chunks=chunk_records)

        nlp_controller = nlpControllers(
            vectordb_client=request.app.vectordb_client,
            embedding_client=request.app.embedding_client,
        )
        try:
            base_id = int(time.time() * 1000)
            index_result = nlp_controller.index_vector_db(
                project=project,
                chunks=chunk_records,
                chunks_ids=[base_id + i for i in range(len(chunk_records))],
                do_reset=False,
            )
            vectors_indexed = len(chunk_records) if index_result else 0
        except Exception as e:
            logger.error(f"❌ VectorDB indexing failed: {e}")
            vectors_indexed = 0

        logger.info(f"✅ Upload done: {chunks_inserted} chunks, {vectors_indexed} vectors")
        return JSONResponse(status_code=200, content={
            "signal": "File uploaded and processed successfully",
            "filename": file.filename,
            "project_id": project_id,
            "file_id": str(asset_record.id),
            "chunks_inserted": chunks_inserted,
            "vectors_indexed_count": vectors_indexed,
            "status": "ok",
        })

    except Exception as e:
        logger.error(f"❌ Professor upload error: {e}")
        return JSONResponse(status_code=500, content={"signal": "Upload failed", "error": str(e)})
