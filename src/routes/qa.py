import asyncio
from collections import defaultdict
import logging
import re
import random
from typing import Dict, List, Optional, TypedDict

from fastapi import APIRouter, HTTPException, Request

from .schemes.qa import (
    QA_enhancement_questions,
    QA_Enhancement_resonce,
    search_QA_Enhancment_request,
)

from controllers.nlpControllers import nlpControllers
from controllers.retrieval import select_balanced_chunks_score_aware
from models.project_model import ProjectModel
from models.chunk_model import chunkModel
from helper.config import get_settings

logger = logging.getLogger("uvicorn.error")

qa_router = APIRouter(
    prefix="/api/v1/data",
    tags=["data"]
)


# ===== HELPER FUNCTIONS =====




def extract_clean_topic(query: str) -> str:
    """Extract clean topic for better VectorDB search."""
    clean_query = query.lower()

    patterns_to_remove = [
        r'\b(?:make|generate|create|give|write|tell|show)\s*(?:me)?\s*',
        r'\b\d+\s*(?:mcq|questions?)\s*',
        r'\b(?:mcq|questions?)\s*',
        r'\babout\s*',
        r'\bon\s*',
        r'\brelated\s*to\s*',
        r'\bto\s*me\s*',
        r'\bof\s*'
    ]

    for pattern in patterns_to_remove:
        clean_query = re.sub(pattern, ' ', clean_query, flags=re.IGNORECASE)

    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    clean_query = re.sub(r'^[^\w]*|[^\w]*$', '', clean_query)

    if len(clean_query) < 3:
        return query

    return clean_query





def _compute_difficulty_counts(
    num_questions: int, percentages: Dict[str, int]
) -> Dict[str, int]:
    """
    Convert difficulty percentages to actual counts for a given total.
    Last key absorbs rounding remainder.
    """
    if num_questions <= 0:
        return {k: 0 for k in percentages}
    keys = list(percentages.keys())
    counts: Dict[str, int] = {}
    allocated = 0
    for key in keys[:-1]:
        cnt = round(num_questions * percentages[key] / 100)
        counts[key] = cnt
        allocated += cnt
    counts[keys[-1]] = max(0, num_questions - allocated)
    return counts


def _build_task_list(
    type_counts: Dict[str, int],
    difficulty_counts: Dict[str, int],
) -> List[tuple]:
    """
    Build the full (qtype, difficulty, count) task list by distributing
    difficulty slots across question types proportionally.

    Computes difficulty on the TOTAL first (already done by caller), then
    assigns each difficulty's quota across types by their share of remaining
    questions. This ensures the global difficulty percentages are respected
    regardless of how many types there are or how small each type count is.

    Returns list of (question_type, difficulty, count) tuples.
    """
    tasks: List[tuple] = []
    type_list = list(type_counts.keys())
    type_remaining = dict(type_counts)

    for diff, diff_total in difficulty_counts.items():
        if diff_total == 0:
            continue
        # Simple weighted-random assignment per slot to avoid fill-first behavior.
        for _ in range(diff_total):
            eligible = [t for t in type_list if type_remaining[t] > 0]
            if not eligible:
                break
            weights = [type_remaining[t] for t in eligible]
            qtype = random.choices(eligible, weights=weights, k=1)[0]
            tasks.append((qtype, diff, 1))
            type_remaining[qtype] -= 1

    merged_tasks: Dict[tuple, int] = {}
    for qtype, diff, count in tasks:
        key = (qtype, diff)
        merged_tasks[key] = merged_tasks.get(key, 0) + count

    # --- Post-process: ensure final per-type totals exactly match `type_counts` ---

    # Compute assigned totals
    assigned_totals: Dict[str, int] = {q: 0 for q in type_list}
    for (qtype, diff), cnt in list(merged_tasks.items()):
        assigned_totals[qtype] = assigned_totals.get(qtype, 0) + cnt

    # Deltas: positive => needs more, negative => over-assigned
    deltas: Dict[str, int] = {q: type_counts.get(q, 0) - assigned_totals.get(q, 0) for q in type_list}

    over_assigned = {q: -d for q, d in deltas.items() if d < 0}
    under_assigned = {q: d for q, d in deltas.items() if d > 0}

    if over_assigned and under_assigned:
        # Order under-assigned targets: prefer MCQ when present, then by larger requested total
        under_order = sorted(
            under_assigned.keys(),
            key=lambda x: (0 if x == "MCQ" else 1, -type_counts.get(x, 0)),
        )

        # Move counts from over-assigned types to under-assigned types
        for over_q, over_amt in list(over_assigned.items()):
            if over_amt <= 0:
                continue

            # Get difficulties for this over type sorted by largest chunk first
            diffs = sorted(
                [(diff, cnt) for (q, diff), cnt in merged_tasks.items() if q == over_q and cnt > 0],
                key=lambda x: -x[1],
            )

            for diff, cnt in diffs:
                if over_amt <= 0:
                    break
                take = min(cnt, over_amt)

                # decrement from source
                merged_tasks[(over_q, diff)] = merged_tasks.get((over_q, diff), 0) - take
                if merged_tasks[(over_q, diff)] <= 0:
                    del merged_tasks[(over_q, diff)]

                over_amt -= take

                # give to the first under-assigned target in preferred order
                for target in under_order:
                    need = under_assigned.get(target, 0)
                    if need <= 0:
                        continue
                    move = min(need, take)
                    merged_tasks[(target, diff)] = merged_tasks.get((target, diff), 0) + move
                    under_assigned[target] -= move
                    take -= move
                    if take <= 0:
                        break

            over_assigned[over_q] = over_amt

    # Clean up zero/negative entries and finalize
    final_tasks: Dict[tuple, int] = {}
    for (qtype, diff), cnt in merged_tasks.items():
        if cnt > 0:
            final_tasks[(qtype, diff)] = final_tasks.get((qtype, diff), 0) + cnt

    # Final safeguard: if tiny mismatches remain (rounding artifacts), fix deterministically
    # by adding/removing from the first difficulty key while preserving totals.
    assigned_final: Dict[str, int] = {q: 0 for q in type_list}
    for (qtype, diff), cnt in final_tasks.items():
        assigned_final[qtype] = assigned_final.get(qtype, 0) + cnt

    # pick a canonical difficulty to add/remove when needed (preserve insertion order of difficulty_counts)
    if difficulty_counts:
        canonical_diff = next(iter(difficulty_counts.keys()))
    else:
        canonical_diff = "medium"

    for q in type_list:
        need = type_counts.get(q, 0) - assigned_final.get(q, 0)
        if need == 0:
            continue
        if need > 0:
            # add missing slots to canonical difficulty
            final_tasks[(q, canonical_diff)] = final_tasks.get((q, canonical_diff), 0) + need
        else:
            # remove surplus from this type starting from largest-difficulty buckets
            surplus = -need
            diffs = sorted(
                [(d, c) for (qq, d), c in list(final_tasks.items()) if qq == q],
                key=lambda x: -x[1],
            )
            for d, c in diffs:
                if surplus <= 0:
                    break
                take = min(c, surplus)
                final_tasks[(q, d)] = final_tasks.get((q, d), 0) - take
                if final_tasks[(q, d)] <= 0:
                    del final_tasks[(q, d)]
                surplus -= take

    return [(qtype, diff, count) for (qtype, diff), count in final_tasks.items() if count > 0]


def _parse_options(options_raw) -> Optional[List[str]]:
    """Normalize options to a list regardless of what the LLM returns."""
    if options_raw is None:
        return None
    if isinstance(options_raw, list):
        return options_raw
    if isinstance(options_raw, str):
        # split on newlines or semicolons
        parts = re.split(r'\n|;', options_raw)
        return [p.strip() for p in parts if p.strip()]
    return None


class ChunkWithScore(TypedDict):
    text: str
    score: float

async def QA_Enhanced_pipeline(
    request,
    search_request: search_QA_Enhancment_request,
) -> dict:
    """
    Enhanced multi-type, multi-project QA pipeline.

    For each project_id: fetches all indexed chunks and merges context.
    Converts difficulty_levels percentages → counts per question type.
    Fires all (type × difficulty) generation tasks in parallel via asyncio.gather
    (or a single batched LLM call when supported), same as ``/QA_enhance``.

    Returns:
        success → {status, questions, project_ids, type_breakdown, difficulty_breakdown}
        error   → {status: "error", detail: str}
    """
    nlp_controller = nlpControllers(
        vectordb_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
    )
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    chunk_model_instance = await chunkModel.create_instance(db_client=request.app.db_client)
    ollama_provider = request.app.generation_client

    try:
        # ── 1. COLLECT CONTEXT FROM ALL PROJECTS ──
        has_topics = bool(search_request.topics)
        retrieval_mode = "topic_search" if has_topics else "full_project_chunks"
        context_chunks: List[str] = []
        valid_project_ids: List[str] = []
        total_chunks_retrieved = 0
        settings = get_settings()
        max_context_chars = max(1024, int(getattr(settings, "GENERATION_CONTEXT_MAX_CHARS", 20000) or 20000))
        max_context_chunks = max(1, int(getattr(settings, "GENERATION_CONTEXT_MAX_CHUNKS", 8) or 8))

        logger.info(
            f"🔎 Retrieval mode={retrieval_mode}, requested_projects={len(search_request.project_ids)}"
        )
        
        # Read retrieval settings from config
        top_k = max(1, int(getattr(settings, "RETRIEVAL_TOP_K", 20) or 20))
        score_threshold = float(getattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.5) or 0.5)
        
        # Read optional human query instruction
        human_query = (search_request.human_query or "").strip() or None
        if human_query:
            logger.info(f"📝 Human query instruction: '{human_query}'")

        # When topics are provided, collect chunks grouped by topic
        # so we can apply balanced per-topic retrieval later.
        chunks_by_topic: Dict[str, List[ChunkWithScore]] = defaultdict(list)

        for pid in search_request.project_ids:
            project = await project_model.get_project_or_create(pid)
            if not project or project.id is None:
                logger.warning(f"⚠️ Project '{pid}' not found, skipping")
                continue

            if has_topics:
                logger.info(f"📂 {pid} | topics={len(search_request.topics)} | searching...")
                project_found_topics = False
                for topic in search_request.topics:
                    clean_topic = extract_clean_topic(topic)
                    logger.info(f"📝 Cleaned topic: '{topic}' → '{clean_topic}'")
                    try:
                        search_results = nlp_controller.search_vector_db_collection(
                            project=project,
                            text=clean_topic,
                            limit=top_k,
                        )
                    except Exception as e:
                        logger.error(f"❌ VectorDB search failed for topic '{clean_topic}': {e}")
                        continue

                    if search_results:
                        passed, filtered = 0, 0
                        for r in search_results:
                            text = r.text if hasattr(r, "text") else ""
                            score = r.score if hasattr(r, "score") else 0.0
                            if score < score_threshold:
                                filtered += 1
                                continue
                            if text.strip():
                                # Group by topic for balanced retrieval
                                chunks_by_topic[clean_topic].append({
                                    "text": text,
                                    "score": score,
                                })
                                passed += 1
                        logger.info(
                            f"✅ Topic '{clean_topic}': {passed} chunks passed threshold "
                            f"(filtered {filtered} below {score_threshold})"
                        )
                        if passed == 0:
                            logger.warning(
                                f"⚠️ Topic '{clean_topic}': all {len(search_results)} chunks "
                                f"scored below threshold {score_threshold}"
                            )
                        if passed > 0:
                            project_found_topics = True
                    else:
                        logger.warning(f"⚠️ No chunks found for topic '{clean_topic}' in project '{pid}'")
                
                if project_found_topics:
                    valid_project_ids.append(pid)

            else:
                # Retrieve all chunks for this project using pagination.
                chunks: List = []
                page_no = 1
                page_size = 500
                while True:
                    page_chunks = await chunk_model_instance.get_project_chunks(
                        project_id=project.id,
                        page_no=page_no,
                        page_size=page_size,
                    )
                    if not page_chunks:
                        break
                    chunks.extend(page_chunks)
                    if len(page_chunks) < page_size:
                        break
                    page_no += 1

                if chunks:
                    # Deterministic ordering inside each project: doc_id -> chunk_index.
                    ordered_chunks = sorted(
                        chunks,
                        key=lambda chunk: (
                            str(getattr(chunk, "chunk_asset_id", "")),
                            int(getattr(chunk, "chunk_order", 0)),
                        ),
                    )

                    per_doc_chunk_counts = defaultdict(int)
                    for chunk in ordered_chunks:
                        per_doc_chunk_counts[str(getattr(chunk, "chunk_asset_id", ""))] += 1
                        text = chunk.chunk_text if hasattr(chunk, "chunk_text") else ""
                        if text.strip():
                            context_chunks.append(text)

                    total_chunks_retrieved += len(ordered_chunks)
                    valid_project_ids.append(pid)
                    logger.info(
                        f"📂 Project '{pid}': {len(ordered_chunks)} chunks collected, "
                        f"all passed (full_project_chunks mode — no score filtering)"
                    )
                    for doc_id, doc_chunk_count in per_doc_chunk_counts.items():
                        logger.debug(
                            f"📄 Project '{pid}' doc '{doc_id}': {doc_chunk_count} chunks (mode={retrieval_mode})"
                        )
                else:
                    logger.warning(f"⚠️ Project '{pid}' has no indexed documents")

        # ── TOPIC-BALANCED RETRIEVAL (topic_search mode) ──
        # Instead of merging all topic chunks into one pool and shuffling
        # globally (which lets the largest topic dominate), we apply a
        # fair per-topic quota so every topic gets equal representation.
        if has_topics and chunks_by_topic:
            context_chunks = select_balanced_chunks_score_aware(
                chunks_by_topic=dict(chunks_by_topic),
                max_chunks=30,
                redistribute_surplus=True,
            )
            logger.info(
                f"📦 Balanced topic retrieval: {len(chunks_by_topic)} topic(s), "
                f"selected {len(context_chunks)}/{top_k} chunk(s)"
            )
        else:
            # full_project_chunks mode: no topic grouping, use simple shuffle + cap
            logger.info(
                f"📦 Retrieval summary: mode={retrieval_mode}, "
                f"total_chunks={len(context_chunks)}, projects={len(valid_project_ids)}"
            )
            # Shuffle chunks to mix projects before capping
            # This prevents bias toward the first project in full_project_chunks mode
            random.shuffle(context_chunks)
            logger.info(f"🔀 Shuffled {len(context_chunks)} chunks to mix project content")

            # Cap total chunks to top_k for full_project_chunks mode
            if len(context_chunks) > top_k:
                logger.info(
                    f"✂️ Capping context: {len(context_chunks)} → {top_k} chunks (RETRIEVAL_TOP_K)"
                )
                context_chunks = context_chunks[:top_k]

        if not context_chunks:
            return {
                "status": "error",
                "detail": "No indexed documents found in any of the provided projects.",
            }

        context = "\n\n".join(context_chunks)
        logger.info(f"📚 Total context: {len(context_chunks)} chunks from {len(valid_project_ids)} project(s)")

        # ── 2. PARSE TYPE → COUNT MAP ──
        type_counts: Dict[str, int] = {}
        for item in search_request.questions_types:
            for qtype, cnt in item.items():
                type_counts[qtype] = cnt

        # ── 3. COMPUTE DIFFICULTY ON TOTAL FIRST, THEN DISTRIBUTE ──
        total_questions = sum(type_counts.values())
        difficulty_counts = _compute_difficulty_counts(
            total_questions, search_request.difficulty_levels
        )
        logger.info(
            f"📊 Total {total_questions}q → difficulty counts: {difficulty_counts}"
        )

        task_meta: List[tuple] = _build_task_list(type_counts, difficulty_counts)
        logger.info(f"📋 Task list: {task_meta}")

        results_by_task: Dict[tuple, List[dict]] = {}
        batch_generate = getattr(ollama_provider, "generate_batch_by_type_and_difficulty", None)

        if callable(batch_generate):
            logger.info("⚡ Running shared-context batched generation (single model request)")
            batched = await batch_generate(context=context, task_meta=task_meta, human_query=human_query)
            if isinstance(batched, dict) and batched:
                for (qtype, diff), items in batched.items():
                    if not isinstance(items, list):
                        continue
                    results_by_task[(qtype, str(diff).lower())] = items
            else:
                logger.warning("⚠️ Batched generation returned no usable groups; falling back to legacy task calls")

        if not results_by_task:
            tasks = [
                ollama_provider.generate_by_type_and_difficulty(
                    context=context,
                    question_type=qtype,
                    difficulty=diff.lower(),
                    num_questions=count,
                    human_query=human_query,
                )
                for qtype, diff, count in task_meta
            ]

            logger.info(f"⚡ Launching {len(tasks)} parallel generation tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (qtype, diff, _), result in zip(task_meta, results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Task [{qtype}/{diff}] failed: {result}")
                    continue
                results_by_task[(qtype, str(diff).lower())] = result

        # ── 4. AGGREGATE RESULTS ──
        all_questions: List[dict] = []
        type_breakdown: Dict[str, int] = {}
        difficulty_breakdown: Dict[str, int] = {}

        for qtype, diff, expected_count in task_meta:
            normalized_diff = str(diff).lower()
            result = results_by_task.get((qtype, normalized_diff), [])

            # ✂️ trim to exactly what was requested
            trimmed = result[:expected_count]

            for q in trimmed:
                q["question_type"] = qtype
                q["difficulty"] = normalized_diff
            all_questions.extend(trimmed)
            type_breakdown[qtype] = type_breakdown.get(qtype, 0) + len(trimmed)
            difficulty_breakdown[normalized_diff] = difficulty_breakdown.get(normalized_diff, 0) + len(trimmed)

        if not all_questions:
            return {"status": "error", "detail": "LLM failed to generate any questions."}

        # Group output questions by type while preserving relative order within each type.
        question_type_order = {"MCQ": 0, "TrueFalse": 1, "Written": 2}
        all_questions = sorted(
            all_questions,
            key=lambda q: question_type_order.get(q.get("question_type"), len(question_type_order)),
        )

        status = "ok" if len(all_questions) == search_request.questions_number else "partial"
        logger.info(f"✅ Enhanced pipeline: {len(all_questions)} questions from {len(valid_project_ids)} project(s)")

        return {
            "status": status,
            "questions": all_questions,
            "project_ids": valid_project_ids,
            "type_breakdown": type_breakdown,
            "difficulty_breakdown": difficulty_breakdown,
        }

    except Exception as e:
        logger.error(f"❌ QA_Enhanced_pipeline error: {e}")
        return {"status": "error", "detail": f"Pipeline failed: {str(e)}"}



# ===== ENHANCED ENDPOINT =====

@qa_router.post("/QA_enhance", response_model=QA_Enhancement_resonce)
async def generate_QA(
    request: Request,
    search_request: search_QA_Enhancment_request,
):
    """Generate multi-type questions across multiple projects with difficulty percentage distribution."""
    result = await QA_Enhanced_pipeline(request=request, search_request=search_request)

    if result["status"] == "error":
        detail = result.get("detail", "Question generation failed")
        status_code = 404 if "no indexed" in detail.lower() else 500
        raise HTTPException(status_code=status_code, detail=detail)

    questions = []
    for q in result.get("questions", []):
        try:
            qtype = q.get("question_type", "MCQ")
            questions.append(QA_enhancement_questions(
                question=q.get("question", ""),
                question_type=qtype,
                difficulty=q.get("difficulty", "medium"),
                options=_parse_options(
                    q.get("options") or q.get("choices") or q.get("answers")
                ),
                correct_answer=(
                    q.get("correct_answer") or q.get("correct")
                ) if qtype != "Written" else None,
                answer=(
                    q.get("answer") or q.get("model_answer") or q.get("reference_answer")
                ) if qtype == "Written" else None,
                explanation=q.get("explanation"),
            ))
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse question: {e}")

    return QA_Enhancement_resonce(
        questions=questions,
        total=len(questions),
        status=result["status"],
        project_ids=result.get("project_ids", search_request.project_ids),
        type_breakdown=result.get("type_breakdown", {}),
        difficulty_breakdown=result.get("difficulty_breakdown", {}),
        message=(
            f"Generated {len(questions)} questions across "
            f"{len(result.get('project_ids', []))} project(s)"
        ),
    )
