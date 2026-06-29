import json
import re
from models.submission import QuestionItem
from helper.grading_prompts import GENERAL_WRITTEN_GRADING_PROMPT, SPECIFIC_WRITTEN_GRADING_PROMPT, GRADING_SYSTEM_PROMPT
from helper.config import get_settings
from stores.llm.LLMfactory import LLMPROVIDEFACTORY

class WrittenGrader:
    def __init__(self, llm_provider=None):
        if llm_provider is None:
            settings = get_settings()
            self.llm_provider = LLMPROVIDEFACTORY(settings).create(settings.GENERATION_BACKEND)
        else:
            self.llm_provider = llm_provider

    async def grade(self, question: QuestionItem) -> dict:
        if question.instructorCriteria and len(question.instructorCriteria) > 0:
            criteria_lines = []
            for c in question.instructorCriteria:
                criteria_lines.append(f"- {c.criteria} (Weight: {c.weight})")
            criteria_text = "\\n".join(criteria_lines)
            
            prompt = SPECIFIC_WRITTEN_GRADING_PROMPT.format(
                question_text=question.questionText,
                max_mark=question.mark,
                expected_answer=question.questionAnswer,
                student_answer=question.studentAnswer,
                criteria_text=criteria_text
            )
        else:
            prompt = GENERAL_WRITTEN_GRADING_PROMPT.format(
                question_text=question.questionText,
                max_mark=question.mark,
                expected_answer=question.questionAnswer,
                student_answer=question.studentAnswer
            )
            
        full_prompt = f"{GRADING_SYSTEM_PROMPT}\n\n{prompt}"
        response = await self.llm_provider.generate_response(full_prompt)
        
        try:
            return self._parse_json_object(response)
        except Exception:
            return {
                "feedback": "Failed to parse grading result.",
                "estimatedScore": 0.0
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
