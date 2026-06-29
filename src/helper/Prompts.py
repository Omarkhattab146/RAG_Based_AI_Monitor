import logging
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableLambda

logger = logging.getLogger("uvicorn.error")


# ============================================================================
# SHARED PROMPT CONSTANTS — Consolidated to reduce token duplication
# ============================================================================

_DIFFICULTY_DEFINITION = """DIFFICULTY DEFINITION:
- EASY: direct recall, definition-based
- MEDIUM: relationship/behavior understanding
- HARD: reasoning, comparison, multi-step thinking"""

_JSON_CONTRACT_CORE = """- Output valid JSON array only; no text before/after JSON.
- Use double quotes only and no trailing commas."""

_CRITICAL_RULES = """CRITICAL:
- Use only the provided context.
- No external knowledge or invented facts.
- Use only definitions stated in the text.
- Do not infer beyond the PDF.
- Do not mix similar concepts.
- Generate a question only if fully answerable from explicit context."""

_QUESTION_OBJECT_SCHEMA = """{
  "question": "string",
  "question_type": "MCQ" | "TrueFalse" | "Written",
  "difficulty": "easy" | "medium" | "hard",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."] | ["True", "False"] | null,
  "correct_answer": "A" | "B" | "C" | "D" | "True" | "False" | null,
    "answer": "full sentence" | null,
  "explanation": "string"
}"""



def _get_batch_type_rules(question_type: str) -> str:
    """Return type-specific output constraints for batched generation directives."""
    if question_type == "MCQ":
        return (
            "- question_type must be \"MCQ\".\n"
            "- options must be exactly 4 array items formatted: \"A) option text\", \"B) option text\", \"C) option text\", \"D) option text\".\n"
            "- correct_answer must be one of: A, B, C, D.\n"
            "- answer must be null.\n"
            "- explanation is required.\n"
            "- Only one correct answer is allowed.\n"
            "- Distractors must be plausible, same domain, and reflect common confusions."
        )
    if question_type == "TrueFalse":
        return (
            "- question_type must be \"TrueFalse\".\n"
            "- options must be exactly [\"True\", \"False\"].\n"
            "- correct_answer must be \"True\" or \"False\".\n"
            "- answer must be null.\n"
            "- explanation is required.\n"
            "- Use declarative statements only (must not end with '?').\n"
            "- Keep True/False balanced when possible."
        )
    return (
        "- question_type must be \"Written\".\n"
        "- options must be null.\n"
        "- correct_answer must be null.\n"
        "- answer must be a clear full sentence grounded in the provided context.\n"
        "- explanation is required."
    )


def build_batch_generation_prompt(
    context: str,
    directives: List[Dict[str, Any]],
    human_query: Optional[str] = None,
) -> str:
    """Build a single batched generation prompt that reuses the same context for all directives."""
    logger.info(f"✅ Built batch prompt: {len(directives)} directive(s)")

    directive_blocks: List[str] = []
    for d in directives:
        directive_blocks.append(
            "\n".join(
                [
                    f"Directive index: {d['index']}",
                    f"Question type: {d['question_type']}",
                    f"Difficulty: {d['difficulty']}",
                    f"Required question count: {d['num_questions']}",
                    _get_batch_type_rules(d["question_type"]),
                ]
            )
        )

    directives_text = "\n\n".join(directive_blocks)
    return f"""You generate university-level assessment items.

{_DIFFICULTY_DEFINITION}

{_CRITICAL_RULES}

Output schema for EACH question object (must be exact):
{_QUESTION_OBJECT_SCHEMA}

You must return one JSON object with this exact top-level shape (batch wrapper):
{{
  "groups": [
    {{
      "index": 0,
      "question_type": "MCQ",
      "difficulty": "easy",
            "questions": [
                {{
                    "question": "What does FastAPI provide for web APIs?",
                    "question_type": "MCQ",
                    "difficulty": "easy",
                    "options": [
                        "A) High performance async framework",
                        "B) Relational database engine",
                        "C) Frontend CSS toolkit",
                        "D) Operating system utility"
                    ],
                    "correct_answer": "A",
                    "answer": null,
                    "explanation": "FastAPI is a Python web framework focused on API performance and developer productivity."
                }}
            ]
    }}
  ]
}}

JSON CONTRACT:
{_JSON_CONTRACT_CORE}


GLOBAL QUALITY RULES:
- Cover different concepts and avoid repeats.
- If context is insufficient for an item, omit that item instead of guessing.
- Never reference the source of information in generated questions or answers (e.g., "according to the context", "based on the text").
- Ensure correct answers are randomly and evenly distributed across all options (A, B, C, D) to avoid positional bias.

SELF-CHECK BEFORE FINAL OUTPUT:
- Verify each answer directly from context.
- Remove duplicate concepts/questions.
- Fix invalid items to match the schema and type-specific rules exactly.

If context is insufficient for a directive, return fewer questions for that directive.


Context/Material:
{context}

Directives:
{directives_text}

Ensure each group keeps the same index/question_type/difficulty as requested.""" + (
    f"""

EXTRA INSTRUCTION FROM USER:
{human_query}
Apply this instruction to ALL generated questions across all directives."""
    if human_query
    else ""
)


def _legacy_single_prompt_adapter(question_type: str, difficulty: str):
    """Compatibility adapter that routes legacy single-generation calls through batch prompt logic."""

    def _to_prompt(payload: Dict[str, Any]) -> str:
        return build_batch_generation_prompt(
            context=str(payload.get("context", "")),
            directives=[
                {
                    "index": 0,
                    "question_type": question_type,
                    "difficulty": str(payload.get("difficulty", difficulty)),
                    "num_questions": int(payload.get("num_questions", 1) or 1),
                }
            ],
        )

    return RunnableLambda(_to_prompt)


# Backward-compatible symbol for legacy imports while internally using batch prompt generation.
get_question_prompt_template = _legacy_single_prompt_adapter

