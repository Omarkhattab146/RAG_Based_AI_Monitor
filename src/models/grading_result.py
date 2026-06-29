from pydantic import BaseModel
from typing import List, Optional

class QuestionFeedback(BaseModel):
    Id: str
    feedback: Optional[str] = None
    estimatedScore: float

class GradingResult(BaseModel):
    QuestionFeedbacks: List[QuestionFeedback]
    weakTopics: List[str]
    sum_score: float
