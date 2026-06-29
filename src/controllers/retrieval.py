"""
Topic-balanced chunk retrieval strategy.

WHY THIS EXISTS
───────────────
The old retrieval flow merged all chunks from every topic into a single pool,
shuffled globally, then capped at ``max_chunks``.  A topic with 50 chunks
drowned out a topic with only 2.  The questions/context were therefore heavily
biased toward the largest topic.

NEW ALGORITHM
─────────────
1. Group chunks **by topic** (they already arrive grouped).
2. Shuffle chunks **inside each topic** independently.
3. Compute a per-topic quota:
       per_topic_limit = max_chunks // topics_count
4. From each topic take  min(per_topic_limit, len(topic_chunks)).
5. Merge selected chunks from all topics.
6. (Optional) redistribute leftover quota from under-filled topics to
   over-filled ones so we don't waste budget.
7. Final shuffle of the merged result.

EDGE CASES
──────────
• max_chunks <= 0          → return empty list
• topics_count == 0        → return empty list
• empty / None collections → skipped silently
• single topic             → behaves like plain cap
• per_topic_limit == 0     → each topic still gets at least 1 chunk when
                             max_chunks >= topics_count (remainder distribution)
• integer division remainder is distributed round-robin across topics so no
  budget is silently discarded.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger("uvicorn.error")


from typing import Dict, List


def select_balanced_chunks_score_aware(
    chunks_by_topic: Dict[str, List[dict]],
    max_chunks: int,
    *,
    redistribute_surplus: bool = True,
) -> List[str]:
    """
    Fair topic-balanced retrieval using retrieval scores.

    Features
    --------
    ✅ No shuffling
    ✅ Highest-score chunks selected first
    ✅ Fair per-topic quota
    ✅ Surplus redistributed round-robin
    ✅ Final ranking by score
    ✅ Detailed retrieval diagnostics for testing
    """

    if max_chunks <= 0:
        logger.warning("⚠️ max_chunks <= 0, returning empty context")
        return []

    active_topics = {
        topic: chunks
        for topic, chunks in (chunks_by_topic or {}).items()
        if chunks
    }

    if not active_topics:
        logger.warning("⚠️ No active topics found")
        return []

    # ------------------------------------------------------------------
    # Sort each topic by score descending
    # ------------------------------------------------------------------
    sorted_topics = {
        topic: sorted(
            chunks,
            key=lambda x: x["score"],
            reverse=True,
        )
        for topic, chunks in active_topics.items()
    }

    logger.info(
        f"📊 Balanced Retrieval Started | "
        f"topics={len(sorted_topics)} | "
        f"max_chunks={max_chunks}"
    )

    for topic, chunks in sorted_topics.items():
        highest = chunks[0]["score"]
        lowest = chunks[-1]["score"]

        logger.info(
            f"📚 Topic '{topic}' "
            f"| candidates={len(chunks)} "
            f"| score_range={highest:.3f} → {lowest:.3f}"
        )

    # ------------------------------------------------------------------
    # Quota calculation
    # ------------------------------------------------------------------
    topics_count = len(sorted_topics)

    base_quota = max_chunks // topics_count
    remainder = max_chunks % topics_count

    topic_names = sorted(sorted_topics.keys())

    quotas = {}

    for idx, topic in enumerate(topic_names):
        quotas[topic] = base_quota + (1 if idx < remainder else 0)

    logger.info(
        f"⚖️ Quota Distribution "
        f"| base_quota={base_quota} "
        f"| remainder={remainder}"
    )

    for topic in topic_names:
        logger.info(
            f"   └─ {topic}: quota={quotas[topic]}"
        )

    # ------------------------------------------------------------------
    # First pass selection
    # ------------------------------------------------------------------
    selected = {}
    surplus = 0

    for topic in topic_names:
        available = sorted_topics[topic]
        quota = quotas[topic]

        take = min(quota, len(available))

        selected[topic] = available[:take]

        unused = quota - take
        surplus += unused

        if take > 0:
            selected_scores = [c["score"] for c in selected[topic]]

            logger.info(
                f"✅ Topic '{topic}' "
                f"| selected={take}/{len(available)} "
                f"| selected_score_range="
                f"{max(selected_scores):.3f} → {min(selected_scores):.3f}"
            )
        else:
            logger.warning(
                f"⚠️ Topic '{topic}' received 0 chunks"
            )

        if unused > 0:
            logger.info(
                f"   ↳ unused quota={unused}"
            )

    # ------------------------------------------------------------------
    # Surplus redistribution
    # ------------------------------------------------------------------
    if redistribute_surplus and surplus > 0:

        candidates = [
            t
            for t in topic_names
            if len(selected[t]) < len(sorted_topics[t])
        ]

        logger.info(
            f"♻️ Starting surplus redistribution "
            f"| surplus={surplus} "
            f"| eligible_topics={len(candidates)}"
        )

        while surplus > 0 and candidates:

            changed = False

            for topic in candidates:

                if surplus <= 0:
                    break

                already_selected = len(selected[topic])

                if already_selected < len(sorted_topics[topic]):

                    added_chunk = sorted_topics[topic][already_selected]

                    selected[topic].append(added_chunk)

                    surplus -= 1
                    changed = True

                    logger.debug(
                        f"➕ Surplus allocated "
                        f"| topic='{topic}' "
                        f"| score={added_chunk['score']:.3f} "
                        f"| remaining_surplus={surplus}"
                    )

            if not changed:
                logger.info(
                    "ℹ️ Redistribution stopped "
                    "(no topic can consume more chunks)"
                )
                break

    # ------------------------------------------------------------------
    # Final topic summary
    # ------------------------------------------------------------------
    logger.info("📦 Final Topic Distribution")

    for topic in topic_names:

        topic_scores = [
            c["score"]
            for c in selected[topic]
        ]

        if topic_scores:
            logger.info(
                f"   └─ {topic}: "
                f"{len(selected[topic])} chunk(s) "
                f"| score_range="
                f"{max(topic_scores):.3f} → {min(topic_scores):.3f} "
                f"| avg_score="
                f"{sum(topic_scores)/len(topic_scores):.3f}"
            )

    # ------------------------------------------------------------------
    # Merge all selected chunks
    # ------------------------------------------------------------------
    merged = []

    for topic in topic_names:
        merged.extend(selected[topic])

    # ------------------------------------------------------------------
    # Global ranking by score
    # ------------------------------------------------------------------
    merged.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    all_scores = [c["score"] for c in merged]

    if all_scores:

        logger.info(
            f"🎯 Final Context Summary "
            f"| chunks={len(merged)}/{max_chunks} "
            f"| score_range="
            f"{max(all_scores):.3f} → {min(all_scores):.3f} "
            f"| avg_score="
            f"{sum(all_scores)/len(all_scores):.3f}"
        )

        logger.info(
            "🏆 Top 10 chunk scores: "
            + ", ".join(
                f"{c['score']:.3f}"
                for c in merged[:10]
            )
        )

        logger.info(
            "🔻 Lowest 10 chunk scores: "
            + ", ".join(
                f"{c['score']:.3f}"
                for c in merged[-10:]
            )
        )

    logger.info(
        f"✅ Balanced retrieval completed "
        f"| selected={len(merged)} chunk(s)"
    )

    return [chunk["text"] for chunk in merged]


def select_balanced_chunks(
    chunks_by_topic: Dict[str, List[str]],
    max_chunks: int,
    *,
    shuffle_seed: Optional[int] = None,
    final_shuffle: bool = True,
    redistribute_surplus: bool = True,
) -> List[str]:
    """Return at most *max_chunks* chunks with **fair per-topic distribution**.

    Parameters
    ----------
    chunks_by_topic:
        Mapping of ``topic_name → list[chunk_text]``.
        Topics with ``None`` or empty lists are silently ignored.
    max_chunks:
        Hard upper limit on total chunks returned.
    shuffle_seed:
        If provided, seeds the RNG for deterministic (reproducible) shuffling.
        Useful for testing.  ``None`` → non-deterministic.
    final_shuffle:
        Whether to shuffle the merged result after per-topic selection.
    redistribute_surplus:
        When ``True``, any leftover quota from topics that had fewer chunks
        than ``per_topic_limit`` is redistributed to topics that still have
        remaining un-selected chunks.  This maximises total usage of the
        budget while still being fairer than a global shuffle.

    Returns
    -------
    List of selected chunk texts, length ≤ *max_chunks*.

    Examples
    --------
    >>> chunks = {"A": list("a"*50), "B": list("b"*2), "C": list("c"*8)}
    >>> len(select_balanced_chunks(chunks, 50))   # 16+2+8 = 26
    26
    """
    # ── Guard: trivial / degenerate inputs ──────────────────────────────
    if max_chunks <= 0:
        return []

    # Filter out None / empty topics
    active_topics: Dict[str, List[str]] = {
        topic: chunks
        for topic, chunks in (chunks_by_topic or {}).items()
        if chunks  # skips None and []
    }

    if not active_topics:
        return []

    # ── Deterministic RNG when a seed is provided ───────────────────────
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else random.Random()

    # ── 1. Shuffle inside each topic ────────────────────────────────────
    shuffled_topics: Dict[str, List[str]] = {}
    for topic, chunks in active_topics.items():
        copy = list(chunks)  # don't mutate the caller's data
        rng.shuffle(copy)
        shuffled_topics[topic] = copy

    topics_count = len(shuffled_topics)

    # ── 2. Compute per-topic quota ──────────────────────────────────────
    #
    #   base_quota  = max_chunks // topics_count   (integer floor)
    #   remainder   = max_chunks  % topics_count   (extra slots to hand out)
    #
    # The remainder is distributed one-per-topic to the first `remainder`
    # topics (alphabetical order for determinism).
    base_quota = max_chunks // topics_count
    remainder  = max_chunks % topics_count

    # Stable iteration order: sort topic names so remainder distribution
    # is deterministic regardless of dict insertion order.
    sorted_topic_names = sorted(shuffled_topics.keys())

    topic_quotas: Dict[str, int] = {}
    for idx, topic in enumerate(sorted_topic_names):
        extra = 1 if idx < remainder else 0
        topic_quotas[topic] = base_quota + extra

    logger.info(
        f"📊 Balanced retrieval: {topics_count} topic(s), "
        f"max_chunks={max_chunks}, base_quota={base_quota}, remainder={remainder}"
    )

    # ── 3. Select chunks per topic (first pass) ────────────────────────
    selected: Dict[str, List[str]] = {}
    surplus_budget = 0  # leftover slots from under-filled topics

    for topic in sorted_topic_names:
        available = shuffled_topics[topic]
        quota = topic_quotas[topic]
        take = min(quota, len(available))
        selected[topic] = available[:take]
        surplus_budget += quota - take  # un-used slots

        logger.debug(
            f"   Topic '{topic}': quota={quota}, available={len(available)}, "
            f"selected={take}"
        )

    # ── 4. Redistribute surplus (optional) ──────────────────────────────
    #
    # Topics that still have un-selected chunks can receive leftover budget.
    # This is a second round-robin pass so the total never exceeds max_chunks
    # but we waste less budget than the naive approach.
    if redistribute_surplus and surplus_budget > 0:
        # Candidates: topics with remaining chunks after the first pass
        candidates = [
            t for t in sorted_topic_names
            if len(selected[t]) < len(shuffled_topics[t])
        ]
        if candidates:
            logger.info(
                f"♻️  Redistributing {surplus_budget} surplus slot(s) "
                f"across {len(candidates)} eligible topic(s)"
            )
            # Round-robin until surplus is exhausted or no candidate can take more
            changed = True
            while surplus_budget > 0 and changed:
                changed = False
                for topic in candidates:
                    if surplus_budget <= 0:
                        break
                    already = len(selected[topic])
                    remaining = shuffled_topics[topic][already:]
                    if remaining:
                        selected[topic].append(remaining[0])
                        surplus_budget -= 1
                        changed = True

    # ── 5. Merge & optionally shuffle ───────────────────────────────────
    merged: List[str] = []
    for topic in sorted_topic_names:
        merged.extend(selected[topic])

    if final_shuffle:
        rng.shuffle(merged)

    total_selected = len(merged)
    logger.info(
        f"✅ Balanced retrieval selected {total_selected}/{max_chunks} chunk(s) "
        f"from {topics_count} topic(s)"
    )
    for topic in sorted_topic_names:
        logger.info(
            f"   └─ Topic '{topic}': {len(selected[topic])} / "
            f"{len(shuffled_topics[topic])} chunk(s)"
        )

    return merged
