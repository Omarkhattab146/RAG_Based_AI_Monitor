import json
import re
from typing import Optional
from models.submission import GradingSubmission
from models.grading_result import GradingResult, QuestionFeedback
from chains.mcq_grader import MCQGrader
from chains.written_grader import WrittenGrader
from helper.grading_prompts import WEAK_TOPICS_EXTRACTION_PROMPT
from helper.config import get_settings
from stores.llm.LLMfactory import LLMPROVIDEFACTORY

class GradingChain:
    def __init__(self):
        settings = get_settings()
        self.llm_provider = LLMPROVIDEFACTORY(settings).create(settings.GENERATION_BACKEND)
        self.mcq_grader = MCQGrader(self.llm_provider)
        self.written_grader = WrittenGrader(self.llm_provider)

    async def run(self, submission: GradingSubmission) -> GradingResult:
        feedbacks = []
        incorrect_feedback_sources = []
        sum_score = 0.0
        
        # Parallelization could be considered here if many questions
        for question in submission.studentAnswers:
            grader_result = {}
            if self._is_objective_question(question.type):
                grader_result = await self.mcq_grader.grade(question, question_type=question.type)
            else:
                grader_result = await self.written_grader.grade(question)

            raw_score = grader_result.get("estimatedScore", 0.0)
            score = self._normalize_score(raw_score, question.mark)
            feedback_text = self._student_feedback(
                grader_result.get("feedback"),
                score,
                question.mark,
                question.type,
                question.questionAnswer,
            )
                
            q_feedback = QuestionFeedback(
                Id=question.id,
                feedback=feedback_text,
                estimatedScore=score,
            )
            feedbacks.append(q_feedback)
            sum_score += q_feedback.estimatedScore
            
            if q_feedback.estimatedScore < question.mark:
                incorrect_feedback_sources.append(
                    self._build_weak_topic_source(
                        question=question,
                        feedback=q_feedback.feedback,
                        score=score,
                    )
                )

        weak_topics = []
        if incorrect_feedback_sources:
            weak_topics = await self.extract_weak_topics(incorrect_feedback_sources)

        return GradingResult(
            QuestionFeedbacks=feedbacks,
            weakTopics=weak_topics,
            sum_score=round(sum_score, 2)
        )

    async def extract_weak_topics(self, feedback_sources: list) -> list:
        combined_text = "\\n\\n".join(feedback_sources)
        prompt = WEAK_TOPICS_EXTRACTION_PROMPT.format(feedbacks=combined_text)
        response = await self.llm_provider.generate_response(prompt)
        
        try:
            topics = self._parse_json_list(response)
            if isinstance(topics, list):
                return self._normalize_weak_topics(topics)
            return []
        except Exception:
            return []

    def _parse_json_list(self, text: str) -> list:
        cleaned = text.strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj

        raise ValueError("Could not parse JSON list from model response")

    def _normalize_weak_topics(self, topics: list) -> list:
        cleaned_topics = []
        seen = set()

        for topic in topics:
            normalized = str(topic or "").strip()
            if not normalized:
                continue

            normalized = re.sub(r"\s+", " ", normalized)
            normalized = normalized.strip("-:;,. ")
            if not normalized:
                continue

            key = normalized.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned_topics.append(normalized)

        return cleaned_topics

    def _build_weak_topic_source(self, question, feedback: Optional[str], score: float) -> str:
        criteria_text = self._format_instructor_criteria(getattr(question, "instructorCriteria", None))
        student_answer = (getattr(question, "studentAnswer", "") or "").strip()
        expected_answer = (getattr(question, "questionAnswer", "") or "").strip()
        question_type = self._normalize_question_type(getattr(question, "type", ""))

        lines = [
            f"Question ID: {getattr(question, 'id', '')}",
            f"Question Type: {question_type}",
            f"Question: {getattr(question, 'questionText', '')}",
            f"Expected Answer: {expected_answer}",
        ]

        if student_answer:
            lines.append(f"Student Answer: {student_answer}")

        if criteria_text:
            lines.append(f"Instructor Criteria: {criteria_text}")

        lines.append(f"{score}/{getattr(question, 'mark', 0)}")

        if feedback and str(feedback).strip():
            lines.append(f"Grading Feedback: {str(feedback).strip()}")

        return "\\n".join(lines)

    def _format_instructor_criteria(self, instructor_criteria) -> str:
        if not instructor_criteria:
            return ""

        parts = []
        for criteria in instructor_criteria:
            criteria_text = str(getattr(criteria, "criteria", "")).strip()
            if not criteria_text:
                continue
            parts.append(criteria_text)

        return "; ".join(parts)

    def _normalize_score(self, score, max_mark: float) -> float:
        try:
            parsed = float(score)
        except Exception:
            parsed = 0.0
        bounded = max(0.0, min(parsed, float(max_mark)))
        return round(bounded, 2)

    def _is_objective_question(self, question_type: str) -> bool:
        normalized = self._normalize_question_type(question_type)
        return normalized in {"MCQ", "TRUEFALSE", "TF", "T&F", "TRUE/FALSE"}

    def _normalize_question_type(self, question_type: str) -> str:
        return str(question_type or "").strip().upper().replace(" ", "")

    def _student_feedback(
        self,
        feedback: str,
        score: float,
        max_mark: float,
        question_type: str,
        expected_answer: str,
    ) -> Optional[str]:
        # Suppress feedback text for fully correct objective questions
        if self._is_objective_question(question_type) and score >= float(max_mark):
            return None

        # If the model returned a specific feedback string, use it verbatim
        if feedback and str(feedback).strip():
            return str(feedback).strip()

        # Fallback: return a minimal score-only string without canned advice
        return f"{score}/{max_mark}"
