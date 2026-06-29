from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict
from enum import Enum

# ✨ Difficulty Levels
class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ✨ Question Types
class QuestionType(str, Enum):
    MCQ = "MCQ"
    TRUE_FALSE = "TrueFalse"
    WRITTEN = "Written"


class search_QA_Enhancment_request(BaseModel):
    project_ids: List[str]
    topics: Optional[List[str]] = None
    human_query: Optional[str] = None
    questions_number: int = 10
    questions_types: List[Dict[str, int]] = Field(
        default_factory=lambda: [{"MCQ": 10}]
    )
    difficulty_levels: Dict[str, int] = Field(
        default_factory=lambda: {"Easy": 30, "Medium": 60, "Hard": 10}
    )

    @root_validator(pre=True)
    def normalize_and_validate(cls, values):
        # Normalize difficulty keys to lowercase while preserving insertion order
        diff = values.get("difficulty_levels")
        if isinstance(diff, dict):
          new_diff: Dict[str, int] = {}
          for k, v in diff.items():
            if not isinstance(k, str):
              continue
            new_diff[k.lower()] = int(v)
          if new_diff:
            values["difficulty_levels"] = new_diff
    
        # Validate and possibly auto-fix questions_types vs questions_number
        qnum = int(values.get("questions_number", 0) or 0)
        qtypes = values.get("questions_types")
        if qtypes is None:
          qtypes = []
    
        # Normalize qtypes into list of dicts and compute sum
        total = 0
        normalized_qtypes: List[Dict[str, int]] = []
        for item in qtypes:
          if not isinstance(item, dict):
            continue
          for k, v in item.items():
            try:
              cnt = int(v)
            except Exception:
              cnt = 0
            normalized_qtypes.append({str(k): cnt})
            total += cnt
    
        # If sum > questions_number -> invalid per requirement
        if total > qnum:
          raise ValueError(f"Sum of questions_types ({total}) exceeds questions_number ({qnum})")
    
        # If sum < questions_number -> auto-add remaining to MCQ
        if total < qnum:
          remaining = qnum - total
          # try to add to existing MCQ entry
          mcq_added = False
          for d in normalized_qtypes:
            if "MCQ" in d:
              d["MCQ"] += remaining
              mcq_added = True
              break
          if not mcq_added:
            normalized_qtypes.append({"MCQ": remaining})
    
        # Ensure at least one type exists
        if not normalized_qtypes:
          normalized_qtypes = [{"MCQ": qnum or 10}]
    
        values["questions_types"] = normalized_qtypes
        return values


class QA_enhancement_questions(BaseModel):
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None


class QA_Enhancement_resonce(BaseModel):
    questions: List[QA_enhancement_questions]
    total: int
    status: str
    project_ids: List[str]
    message: Optional[str] = None
    type_breakdown: Dict[str, int] = Field(default_factory=dict)
    difficulty_breakdown: Dict[str, int] = Field(default_factory=dict)



# ✅ Documentation Examples (اختياري - للتوضيح فقط)
"""
Example Request:
{
  "Query": "Make 10 mcq questions about Python",
  "mode": "auto",
  "num_questions": 10,
  "limit": 10,
  "difficulty_distribution": {
    "easy": 2,
    "medium": 5,
    "hard": 3
  }
}

Example Response:
{
  "questions": [
    {
      "question": "What is Python?",
      "options": ["A) Programming language", "B) Snake", "C) Framework", "D) Database"],
      "correct_answer": "A",
      "explanation": "Python is a high-level programming language",
      "difficulty": "easy"
    }
  ],
  "total": 10,
  "status": "ok",
  "message": "Generated 10 questions using auto mode with difficulty levels"
}
"""