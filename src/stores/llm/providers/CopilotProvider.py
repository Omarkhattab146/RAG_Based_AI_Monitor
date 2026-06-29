import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from helper.Prompts import build_batch_generation_prompt
from helper.config import get_settings
from stores.llm.github_models_config import get_chat_llm


logger = logging.getLogger("uvicorn.error")


class CopilotRoles(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class CopilotProvider:
    """LLM provider that uses GitHub Models via ChatOpenAI."""

    def __init__(self):
        self.settings = get_settings()
        self.model = (
            getattr(self.settings, "COPILOT_MODEL", None)
            or getattr(self.settings, "GITHUB_MODELS_MODEL", None)
            or getattr(self.settings, "GENERATION_MODEL_ID", None)
            or "gpt-4o"
        )
        self.num_parallel = max(1, int(getattr(self.settings, "COPILOT_NUM_PARALLEL", 4) or 4))
        self.context_length = max(256, int(getattr(self.settings, "COPILOT_CONTEXT_LENGTH", 4096) or 4096))
        self.max_completion_tokens = getattr(self.settings, "GENERATION_DEFAULT_MAX_TOKENS", None)
        self._generation_semaphore = asyncio.Semaphore(self.num_parallel)
        self._generation_lock = asyncio.Lock()
        raw_gap = getattr(self.settings, "COPILOT_REQUEST_GAP_SECONDS", "NOT_FOUND")
        logger.info(f"🔧 DIAG | COPILOT_REQUEST_GAP_SECONDS from settings: {raw_gap}")
        self._min_request_gap_seconds = float(
            getattr(self.settings, "COPILOT_REQUEST_GAP_SECONDS", 20.0)
        )
        logger.info(f"🔧 DIAG | request_gap_seconds: {self._min_request_gap_seconds}")
        self._last_request_finished_at = 0.0
        self.enums = CopilotRoles

        try:
            self.llm = self._build_llm(model_name=self.model)
            logger.info(
                f"CopilotProvider initialized with model: {self.model}, "
                f"num_parallel: {self.num_parallel}, context_length: {self.context_length}"
            )
        except Exception as exc:
            logger.error(f"Failed to initialize CopilotProvider: {exc}")
            raise

    def _build_llm(self, model_name: Optional[str] = None) -> ChatOpenAI:
        return get_chat_llm(
            model_name=model_name or self.model,
            max_tokens=self.max_completion_tokens,
            timeout=getattr(self.settings, "COPILOT_TIMEOUT", 60),
        )

    def _limit_context(self, context: str) -> str:
        if not context:
            return context
        max_chars = self.context_length * 4
        if len(context) <= max_chars:
            return context
        return context[:max_chars]

    def _wait_for_request_gap_sync(self) -> None:
        if self._last_request_finished_at <= 0:
            return

        elapsed = time.monotonic() - self._last_request_finished_at
        if elapsed < self._min_request_gap_seconds:
            time.sleep(self._min_request_gap_seconds - elapsed)

    async def _wait_for_request_gap_async(self) -> None:
        elapsed = time.monotonic() - self._last_request_finished_at
        logger.info(
            f"🔧 DIAG | wait_for_gap: last_finished={self._last_request_finished_at:.2f}, "
            f"elapsed={elapsed:.2f}s, min_gap={self._min_request_gap_seconds:.2f}s"
        )
        if self._last_request_finished_at <= 0:
            return

        if elapsed < self._min_request_gap_seconds:
            await asyncio.sleep(self._min_request_gap_seconds - elapsed)

    def _parse_json_response(self, response_text: str) -> List[Dict]:
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        parsed_payload = None

        try:
            parsed_payload = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        if parsed_payload is None:
            json_match = re.search(r"\[[\s\S]*\]", cleaned)
            if json_match:
                try:
                    parsed_payload = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    parsed_payload = None

        if parsed_payload is None:
            decoder = json.JSONDecoder()
            recovered_items = []
            idx = 0
            while idx < len(cleaned):
                brace_pos = cleaned.find("{", idx)
                if brace_pos == -1:
                    break
                try:
                    obj, end_pos = decoder.raw_decode(cleaned[brace_pos:])
                    if isinstance(obj, dict):
                        recovered_items.append(obj)
                    idx = brace_pos + end_pos
                except json.JSONDecodeError:
                    idx = brace_pos + 1

            if recovered_items:
                parsed_payload = recovered_items

        if parsed_payload is None:
            logger.error("Could not parse JSON response. Full raw response follows:\n%s", response_text)
            return []

        if isinstance(parsed_payload, dict):
            groups = parsed_payload.get("groups")
            if isinstance(groups, list) and groups:
                first_group = groups[0] if isinstance(groups[0], dict) else None
                group_questions = first_group.get("questions") if isinstance(first_group, dict) else None
                if isinstance(group_questions, list):
                    questions = group_questions
                else:
                    questions = []
            elif isinstance(parsed_payload.get("questions"), list):
                questions = parsed_payload["questions"]
            else:
                questions = [parsed_payload]
        elif isinstance(parsed_payload, list):
            questions = parsed_payload
        else:
            logger.error("Parsed payload is neither dict nor list: %s", type(parsed_payload).__name__)
            return []

        normalized_questions: List[Dict] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            q = dict(q)

            if "correct_answer" in q and isinstance(q["correct_answer"], str):
                q["correct_answer"] = q["correct_answer"].rstrip(")").strip()

            if q.get("question_type") == "Written" and "correct_answer" in q:
                del q["correct_answer"]

            normalized_questions.append(q)

        return normalized_questions

    def _parse_batch_json_response(self, response_text: str) -> Dict[int, List[Dict]]:
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        parsed_payload: Optional[Any] = None

        try:
            parsed_payload = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        if parsed_payload is None:
            json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if json_match:
                try:
                    parsed_payload = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    parsed_payload = None

        if not isinstance(parsed_payload, dict):
            logger.error("Could not parse batched JSON response.")
            return {}

        groups = parsed_payload.get("groups")
        if not isinstance(groups, list):
            logger.error("Batched response is missing 'groups' array.")
            return {}

        by_index: Dict[int, List[Dict]] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue

            idx_raw = group.get("index")
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                continue

            questions = group.get("questions")
            if not isinstance(questions, list):
                continue

            normalized_questions = [q for q in questions if isinstance(q, dict)]
            by_index[idx] = normalized_questions

        return by_index

    def _extract_mcq_key(self, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return None
        s = value.strip().upper()
        if s in {"A", "B", "C", "D"}:
            return s
        m = re.match(r"^([ABCD])[\)\.\:\-\s].*", s)
        if m:
            return m.group(1)
        return None

    def _normalize_options(self, options_raw) -> Optional[List[str]]:
        if options_raw is None:
            return None
        if isinstance(options_raw, list):
            cleaned = [str(o).strip() for o in options_raw if str(o).strip()]
            return cleaned or None
        if isinstance(options_raw, str):
            parts = re.split(r"\n|;", options_raw)
            cleaned = [p.strip() for p in parts if p.strip()]
            return cleaned or None
        return None

    def _validate_question(self, q: Dict, expected_type: str, difficulty: str) -> Optional[Dict]:
        question_text = str(q.get("question", "")).strip()
        if not question_text:
            return None

        normalized = {
            "question": question_text,
            "question_type": expected_type,
            "difficulty": difficulty,
        }

        explanation = q.get("explanation")
        if explanation is not None and str(explanation).strip():
            normalized["explanation"] = str(explanation).strip()

        if expected_type == "MCQ":
            options = self._normalize_options(
                q.get("options") or q.get("choices") or q.get("answers")
            )
            if not options or len(options) != 4:
                return None
            valid_labels = ["A)", "B)", "C)", "D)"]
            for idx, opt in enumerate(options):
                if not str(opt).strip().startswith(valid_labels[idx]):
                    return None

            key = self._extract_mcq_key(
                str(q.get("correct_answer") or q.get("correct") or q.get("answer") or "")
            )
            if key is None:
                return None

            normalized["options"] = options
            normalized["correct_answer"] = key
            return normalized

        if expected_type == "TrueFalse":
            answer = str(q.get("correct_answer") or q.get("answer") or "").strip()
            if answer not in {"True", "False"}:
                return None
            normalized["options"] = ["True", "False"]
            normalized["correct_answer"] = answer
            return normalized

        if expected_type == "Written":
            answer = q.get("answer") or q.get("model_answer") or q.get("reference_answer")
            if answer is None or not str(answer).strip():
                return None
            normalized["answer"] = str(answer).strip()
            return normalized

        return None

    def _validate_questions_batch(self, questions: List[Dict], expected_type: str, difficulty: str) -> List[Dict]:
        valid: List[Dict] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            item = self._validate_question(q, expected_type=expected_type, difficulty=difficulty)
            if item is not None:
                valid.append(item)
        return valid

    def set_generation_model(self, model_id: str):
        if not model_id:
            logger.debug("set_generation_model called with empty model_id; keeping current model")
            return

        self.model = model_id
        self.llm = self._build_llm(model_name=model_id)
        logger.info(f"CopilotProvider generation model set to: {self.model}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        logger.debug("set_embedding_model(%s, %s) called on CopilotProvider - no-op", model_id, embedding_size)

    def check_available(self) -> bool:
        try:
            logger.info("🔧 DIAG | check_available() called — will ping API")
            token = getattr(self.settings, "GITHUB_TOKEN", None)
            if not token:
                logger.error("GITHUB_TOKEN is not configured")
                return False

            logger.info("🔧 DIAG | Sending ping request to GitHub Models...")
            test_response = self.llm.invoke("ping")
            logger.info("🔧 DIAG | Ping succeeded")
            return True
        except Exception as exc:
            logger.error(f"GitHub Models not available: {exc}")
            return False

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": prompt,
        }

    def _build_messages(self, prompt: str, chat_history: Optional[list] = None):
        messages = []
        for item in chat_history or []:
            if item is None:
                continue
            if hasattr(item, "content"):
                messages.append(item)
                continue
            if isinstance(item, dict):
                role = str(item.get("role", self.enums.USER.value)).lower()
                text = str(item.get("text") or item.get("content") or "")
                if role == self.enums.SYSTEM.value:
                    messages.append(SystemMessage(content=text))
                elif role in {self.enums.ASSISTANT.value, "assistant", "ai"}:
                    messages.append(AIMessage(content=text))
                else:
                    messages.append(HumanMessage(content=text))

        messages.append(HumanMessage(content=prompt))
        return messages

    def generate_text(
        self,
        prompt: str,
        chat_history: Optional[list] = None,
        max_output_tokens: int = None,
    ):
        try:
            model = self.llm
            if max_output_tokens is not None:
                model = model.bind(max_tokens=max_output_tokens)

            self._wait_for_request_gap_sync()
            result = model.invoke(self._build_messages(prompt, chat_history))
            self._last_request_finished_at = time.monotonic()
            return result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            self._last_request_finished_at = time.monotonic()
            logger.error(f"Error in generate_text: {exc}")
            return f"Error: {str(exc)}"

    async def generate_response(
        self,
        prompt: str,
    ) -> str:
        try:
            async with self._generation_lock:
                model = self.llm
                await self._wait_for_request_gap_async()
                result = await model.ainvoke(prompt)
                self._last_request_finished_at = time.monotonic()
                return result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            self._last_request_finished_at = time.monotonic()
            logger.error(f"Error in generate_response: {exc}")
            return f"Error: {str(exc)}"

    def embed_text(self, text: str, doc_type: str = None):
        raise NotImplementedError("Embedding is not supported by CopilotProvider")

    async def generate_mcq_by_difficulty(
        self,
        context: str,
        topic: str,
        difficulty: str,
        num_questions: int,
    ) -> List[Dict]:
        return await self.generate_by_type_and_difficulty(
            context=context,
            question_type="MCQ",
            difficulty=difficulty,
            num_questions=num_questions,
        )

    async def generate_mcq_questions(
        self,
        context: str,
        topic: str,
        num_questions: int = 10,
        difficulty_distribution: Optional[Dict[str, int]] = None,
    ) -> Dict:
        logger.info(f"MCQ generation (parallel): {num_questions}q about '{topic}'")
        difficulty_distribution = difficulty_distribution or {}

        all_questions = []
        generation_log = {}

        for difficulty, count in difficulty_distribution.items():
            if count <= 0:
                continue

            try:
                result = await self.generate_mcq_by_difficulty(
                    context=context,
                    topic=topic,
                    difficulty=difficulty,
                    num_questions=count,
                )
                all_questions.extend(result)
                generation_log[difficulty] = len(result)
            except Exception as exc:
                logger.error(f"{difficulty} generation failed: {exc}")
                generation_log[difficulty] = 0

        logger.info(f"Sequential done: {len(all_questions)}q | {generation_log}")

        return {
            "status": "ok" if all_questions else "partial",
            "questions": all_questions,
            "total": len(all_questions),
            "distribution": difficulty_distribution,
            "generated": generation_log,
            "message": f"Generated {len(all_questions)} questions (parallel)",
        }

    async def generate_by_type_and_difficulty(
        self,
        context: str,
        question_type: str,
        difficulty: str,
        num_questions: int,
        human_query: Optional[str] = None,
    ) -> List[Dict]:
        try:
            logger.info(f"Generating {num_questions} {question_type}/{difficulty} questions")
            limited_context = self._limit_context(context)
            prompt = build_batch_generation_prompt(
                context=limited_context,
                directives=[
                    {
                        "index": 0,
                        "question_type": question_type,
                        "difficulty": difficulty,
                        "num_questions": num_questions,
                    }
                ],
                human_query=human_query,
            )
            structured_llm = self.llm.bind(response_format={"type": "json_object"})

            async with self._generation_lock:
                await self._wait_for_request_gap_async()
                result = await structured_llm.ainvoke(prompt)
                self._last_request_finished_at = time.monotonic()

            result_text = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json_response(result_text)
            valid = self._validate_questions_batch(
                parsed,
                expected_type=question_type,
                difficulty=difficulty,
            )

            collected: List[Dict] = []
            seen_questions = set()
            for q in valid:
                key = q.get("question", "").strip().lower()
                if not key or key in seen_questions:
                    continue
                seen_questions.add(key)
                collected.append(q)
                if len(collected) >= num_questions:
                    break

            logger.info(
                f"{question_type}/{difficulty} single-attempt: "
                f"parsed={len(parsed)} valid={len(valid)} returned={len(collected)}/{num_questions}"
            )

            if len(collected) < num_questions:
                logger.warning(
                    f"{question_type}/{difficulty}: returning {len(collected)}/{num_questions} valid questions"
                )

            return collected[:num_questions]
        except Exception as exc:
            self._last_request_finished_at = time.monotonic()
            logger.error(f"generate_by_type_and_difficulty [{question_type}/{difficulty}]: {exc}")
            return []

    async def generate_batch_by_type_and_difficulty(
        self,
        context: str,
        task_meta: List[Tuple[str, str, int]],
        human_query: Optional[str] = None,
    ) -> Dict[Tuple[str, str], List[Dict]]:
        if not task_meta:
            return {}

        try:
            directives: List[Dict[str, Any]] = []
            for idx, (qtype, diff, count) in enumerate(task_meta):
                if count <= 0:
                    continue
                directives.append(
                    {
                        "index": idx,
                        "question_type": qtype,
                        "difficulty": str(diff).lower(),
                        "num_questions": int(count),
                    }
                )

            if not directives:
                return {}

            limited_context = self._limit_context(context)
            prompt = build_batch_generation_prompt(limited_context, directives, human_query=human_query)
            structured_llm = self.llm.bind(response_format={"type": "json_object"})

            async with self._generation_lock:
                await self._wait_for_request_gap_async()
                response = await structured_llm.ainvoke(prompt)
                self._last_request_finished_at = time.monotonic()

            response_text = response.content if hasattr(response, "content") else str(response)
            grouped_raw = self._parse_batch_json_response(response_text)

            grouped_valid: Dict[Tuple[str, str], List[Dict]] = {}
            for d in directives:
                idx = d["index"]
                qtype = d["question_type"]
                diff = d["difficulty"]
                needed = d["num_questions"]

                parsed_questions = grouped_raw.get(idx, [])
                valid = self._validate_questions_batch(
                    parsed_questions,
                    expected_type=qtype,
                    difficulty=diff,
                )

                collected: List[Dict] = []
                seen_questions = set()
                for q in valid:
                    key = q.get("question", "").strip().lower()
                    if not key or key in seen_questions:
                        continue
                    seen_questions.add(key)
                    collected.append(q)
                    if len(collected) >= needed:
                        break

                grouped_valid[(qtype, diff)] = collected[:needed]

                logger.info(
                    f"{qtype}/{diff} batched: "
                    f"parsed={len(parsed_questions)} valid={len(valid)} returned={len(collected)}/{needed}"
                )

            return grouped_valid
        except Exception as exc:
            self._last_request_finished_at = time.monotonic()
            logger.error(f"generate_batch_by_type_and_difficulty: {exc}")
            return {}
