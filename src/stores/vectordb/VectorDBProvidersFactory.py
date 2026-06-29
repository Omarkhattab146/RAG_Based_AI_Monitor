from .providers import QuadrantDBProvider
from .vectorDBenums import VectorDBEnums
from controllers.BaseController import BaseController

class VectorDBProvidersFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_db_path(db_name = self.config.VECTOR_DB_PATH)
            return QuadrantDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD
            )
        return None
    
    
    