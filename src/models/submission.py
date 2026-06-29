from pydantic import BaseModel, Field
from typing import List, Optional

class InstructorCriteria(BaseModel):
    criteria: str
    weight: float

class QuestionItem(BaseModel):
    id: str
    questionText: str
    type: str # "MCQ" or "Written"
    mark: float
    options: Optional[List[str]] = None
    studentAnswer: str
    questionAnswer: str
    instructorCriteria: Optional[List[InstructorCriteria]] = None

class GradingSubmission(BaseModel):
    project_Ids: List[str]
    studentAnswers: List[QuestionItem]
