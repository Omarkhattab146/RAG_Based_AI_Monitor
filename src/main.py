from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

from helper.config import get_settings
from routes.grading import router as grading_router
from routes.professor import professor_router
from routes.qa import qa_router
from stores.llm.LLMfactory import LLMPROVIDEFACTORY
from stores.vectordb.VectorDBProvidersFactory import VectorDBProvidersFactory

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.mongodb_connection = AsyncIOMotorClient(settings.MONGO_URI)
    app.db_client = app.mongodb_connection[settings.MONGODB_DATABASE]

    llm_factory = LLMPROVIDEFACTORY(settings)

    app.generation_client = llm_factory.create(settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_factory.create(settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE,
    )

    vector_db_factory = VectorDBProvidersFactory(settings)
    app.vectordb_client = vector_db_factory.create(provider=settings.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()

    logger.info("✅ Application startup complete")
    yield

    app.mongodb_connection.close()
    app.vectordb_client.disconnect()
    logger.info("🛑 Application shutdown complete")


app = FastAPI(lifespan=lifespan)
app.include_router(professor_router)
app.include_router(qa_router)
app.include_router(grading_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"❌ Validation Error for {request.url.path}:")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(await request.body()),
        },
    )
