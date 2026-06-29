from helper.config import get_settings

class BaseDataModel:
    #  Base class for all MongoDB models to share database client
    def __init__(self, db_client: object):
        self.app_settings = get_settings()
        self.db_client = db_client  

    