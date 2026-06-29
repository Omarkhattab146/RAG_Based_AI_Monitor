# Base class with utility methods (file dir setup, random string generation)

# from Data conroller file
from helper.config import get_settings
from fastapi import UploadFile
import os
import random
import string

# Purpose: Common utilities for file & directory management
class BaseController():
    #
    def __init__(self):

        self.app_settings = get_settings()
        # Navigate from controllers → src directory
        self.base_dir = os.path.dirname(os.path.dirname(__file__)) #get parent directory which is the controllers then parent of controllers which is src
        # go to assets then files
        self.file_dir = os.path.join(self.base_dir, "assets", "files") 
        # self.file_dir = self.base_dir + "/assets/files/" # go to assets then files

        # Path to assets/database (where vector DB is stored)
        self.db_dir = os.path.join(self.base_dir, "assets", "database") 

    def generate_random_string(self, length: int = 12): 
        # generate random string of fixed length to be used in file uploaded
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def get_db_path(self, db_name:str):
        # Create directory if it doesn't exist
        database_path = os.path.join(
            self.db_dir,
            db_name
        )

        if not os.path.exists(database_path):
            os.makedirs(database_path)
        
        return database_path
