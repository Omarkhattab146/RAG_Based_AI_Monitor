import json
import re
from models.submission import QuestionItem
from helper.grading_prompts import MCQ_GRADING_PROMPT, GRADING_SYSTEM_PROMPT
from helper.config import get_settings
from stores.llm.LLMfactory import LLMPROVIDEFACTORY

class MCQGrader:
    def __init__(self, llm_provider=None):
        if llm_provider is None:
            settings = get_settings()
            self.llm_provider = LLMPROVIDEFACTORY(settings).create(settings.GENERATION_BACKEND)
        else:
            self.llm_provider = llm_provider

    async def grade(self, question: QuestionItem, question_type: str = "MCQ") -> dict:
        options_text = "\\n".join(question.options) if question.options else "True / False" if str(question_type).upper().replace(" ", "") in {"TRUEFALSE", "TF", "T&F", "TRUE/FALSE"} else "None provided"
        normalized_type = str(question_type or "MCQ").strip()
        
        prompt = MCQ_GRADING_PROMPT.format(
            question_text=question.questionText,
            question_type=normalized_type,
            options=options_text,
            max_mark=question.mark,
            expected_answer=question.questionAnswer,
            student_answer=question.studentAnswer
        )
        
        full_prompt = f"{GRADING_SYSTEM_PROMPT}\n\n{prompt}"
        response = await self.llm_provider.generate_response(full_prompt)
        
        try:
            return self._parse_json_object(response)
        except Exception:
            # Fallback if json parsing fails
            # Simplified generic rule base fallback could go here
            student_answer = (question.studentAnswer or "").strip().lower()
            expected_answer = (question.questionAnswer or "").strip().lower()
            is_match = student_answer == expected_answer
            return {
                "feedback": None if is_match else "Answer is incorrect.",
                "estimatedScore": float(question.mark) if is_match else 0.0
            }

    def _parse_json_object(self, text: str) -> dict:
        cleaned = text.strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj

        raise ValueError("Could not parse JSON object from model response")
