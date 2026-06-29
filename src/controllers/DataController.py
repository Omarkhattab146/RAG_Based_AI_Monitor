# Validates uploaded files (type, size) & generates unique file paths
from .BaseController import BaseController
from fastapi import UploadFile
from .ProjectController import ProjectController
import re
import os

# Purpose: (file validation, path generation)
class DataController(BaseController):
    # forward to making base Conroller file

    def __init__(self):
        super().__init__()
        
    def Validate_uploaded_file(self, file: UploadFile):
        # validate is the file type is acccurate to me
        if file.content_type not in self.app_settings.FILE_ALOWED_EXTENSTIONS:
            return False
        elif file.size > (self.app_settings.FILE_MAX_SIZE * 1024 * 1024):
            return False
        
        return True
    
    def generate_unique_file_path(self, original_filename: str, project_id: str):
        random_filename = self.generate_random_string()
        project_path = ProjectController().get_project_path(project_id=project_id)
    
        clean_filename = re.sub(r'[^\w.]', ' ', original_filename)
        clean_filename = clean_filename.replace(" ", "_")   


        full_path = os.path.join(project_path, random_filename + "_" + clean_filename)
        
        # validate uniqueness in file name
        while os.path.exists(full_path):
            random_filename = self.generate_random_string()
            full_path = os.path.join(project_path, random_filename + "_" + clean_filename)
        
        return full_path, random_filename + "_" + clean_filename


