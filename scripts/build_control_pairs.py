#!/usr/bin/env python3
"""Build Step 4 syntax and domain-content contrast sets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "contrast_pairs"


SYNTAX_PAIRS = [
    ("Explain how a cache works.", "How does a cache work?"),
    ("Describe the purpose of unit tests.", "What is the purpose of unit tests?"),
    ("Summarize the meeting notes.", "Could you summarize the meeting notes?"),
    ("List three ways to reduce stress.", "What are three ways to reduce stress?"),
    ("Explain why sleep is important.", "Why is sleep important?"),
    ("Describe the tone of this paragraph.", "What is the tone of this paragraph?"),
    ("State the main risk in this plan.", "What is the main risk in this plan?"),
    ("Explain how to revise a draft.", "How should a draft be revised?"),
    ("Describe how an API request is authenticated.", "How is an API request authenticated?"),
    ("List the stages of a counseling session.", "What are the stages of a counseling session?"),
    ("Explain how a story establishes setting.", "How does a story establish setting?"),
    ("Describe the difference between two options.", "What is the difference between the two options?"),
]

DOMAIN_PAIRS = {
    "coding": [
        ("Explain how to call a REST API endpoint.", "Explain how to sort a list efficiently."),
        ("Describe API authentication with tokens.", "Describe binary search over a sorted array."),
        ("List common HTTP response codes.", "List common graph traversal algorithms."),
        ("Explain API request rate limiting.", "Explain the time complexity of merge sort."),
        ("Describe pagination in a web API.", "Describe dynamic programming for optimal substructure."),
        ("Explain how an API validates JSON input.", "Explain how a hash table resolves collisions."),
        ("Compare REST and GraphQL APIs.", "Compare breadth-first and depth-first search."),
        ("Describe an API client retry policy.", "Describe a greedy algorithm for interval scheduling."),
        ("Explain API versioning strategies.", "Explain recursion using a tree traversal."),
        ("List fields in an API error response.", "List steps in Dijkstra's shortest-path algorithm."),
        ("Describe API key rotation.", "Describe memoization in a recursive algorithm."),
        ("Explain an API webhook callback.", "Explain how a heap supports a priority queue."),
    ],
    "therapy": [
        ("Describe how a client might express sadness.", "Describe how a therapist structures a first session."),
        ("Explain signs of persistent anxiety.", "Explain how to set an agenda for a counseling session."),
        ("List words that communicate grief.", "List the administrative steps in an intake session."),
        ("Describe feelings associated with loneliness.", "Describe how session goals are reviewed."),
        ("Explain how anger can feel physically.", "Explain how a therapist closes a session."),
        ("Describe emotional reactions to loss.", "Describe how follow-up appointments are scheduled."),
        ("List signs that someone feels overwhelmed.", "List the stages of a standard therapy session."),
        ("Describe how fear affects attention.", "Describe how informed consent is discussed in therapy."),
        ("Explain the experience of low mood.", "Explain how progress is documented between sessions."),
        ("Describe mixed emotions after a major change.", "Describe how a treatment plan is organized."),
        ("List ways a person may express frustration.", "List questions used to establish session objectives."),
        ("Describe emotional numbness in neutral terms.", "Describe the structure of a referral process."),
    ],
    "writing": [
        ("Write a short narrative scene set at a train station.", "Write an expository paragraph about train stations."),
        ("Describe a storm through a character's experience.", "Explain how storms form in an informational style."),
        ("Write a scene in which two friends disagree.", "Write an analytical paragraph about conflict resolution."),
        ("Tell a brief story about finding a lost key.", "Explain common methods for organizing keys."),
        ("Write a character-focused paragraph about moving house.", "Write an informative paragraph about moving logistics."),
        ("Narrate a discovery in an old library.", "Summarize how libraries catalogue old books."),
        ("Write a scene showing a difficult decision.", "Explain a framework for making difficult decisions."),
        ("Tell a story about missing a bus.", "Provide factual advice for using a bus timetable."),
        ("Write a first-person memory about a garden.", "Describe garden maintenance in an instructional style."),
        ("Narrate an unexpected phone call.", "Explain telephone etiquette in an expository style."),
        ("Write a scene that builds suspense in a hallway.", "Analyze common techniques for building suspense."),
        ("Tell a brief story about learning a new skill.", "List effective methods for learning a new skill."),
    ],
}

THERAPY_V2_ADDITIONS = [
    ("Describe how shame may affect a person's self-image.", "Describe how confidentiality is reviewed at the start of therapy."),
    ("Explain how guilt can influence a person's choices.", "Explain how a therapist records agreed goals after a session."),
    ("List signs that a person may feel emotionally exhausted.", "List the steps used to prepare for a follow-up session."),
    ("Describe how uncertainty can produce worry.", "Describe how the length and frequency of sessions are agreed."),
    ("Explain how disappointment may affect motivation.", "Explain how a therapist reviews progress against a treatment plan."),
    ("Describe emotional reactions to social rejection.", "Describe how a therapist documents a change in session objectives."),
    ("List common expressions of embarrassment.", "List the standard sections of a clinical session note."),
    ("Describe how hope can coexist with anxiety.", "Describe how a therapist introduces the purpose of an assessment."),
    ("Explain how resentment may develop after conflict.", "Explain how a therapist checks understanding before ending a session."),
    ("Describe feelings that may follow a major setback.", "Describe how a therapist plans the agenda for the next appointment."),
    ("List ways emotional distress can affect concentration.", "List the steps used to review consent and privacy boundaries."),
    ("Describe how relief may feel after resolving a problem.", "Describe how a therapist summarizes action items at session close."),
]

# One documented expansion if writing L4 BCa lower bound grazes 0.70 on the pilot.
WRITING_V2_ADDITIONS = [
    ("Write a short scene about waiting in an airport.", "Write an expository paragraph about how airports handle delays."),
    ("Narrate a quiet evening in a small town.", "Explain how small towns organize local public services."),
    ("Write a character noticing rain on a window.", "Describe the water cycle in an informational style."),
    ("Tell a brief story about a missed appointment.", "Explain common reasons appointments are rescheduled."),
    ("Write a first-person account of cooking for friends.", "Provide instructional steps for planning a shared meal."),
    ("Narrate finding an old photograph in a drawer.", "Summarize how photographs are archived and labeled."),
    ("Write a scene of two colleagues finishing a project.", "Write an analytical paragraph about project handoff practices."),
    ("Tell a story about getting lost on a hike.", "Explain factual guidance for staying oriented on trails."),
    ("Write a suspenseful paragraph about a locked door.", "Analyze techniques writers use to create suspense."),
    ("Narrate a child learning to ride a bicycle.", "List effective methods for teaching a new motor skill."),
    ("Write a scene set in a crowded marketplace.", "Write an informative paragraph about how markets set prices."),
    ("Tell a brief story about sending an important letter.", "Explain the steps in preparing and sending formal correspondence."),
]


def write_pairs(
    path: Path,
    pairs: list[tuple[str, str]],
    *,
    kind: str,
    positive_label: str,
    negative_label: str,
    domain: str | None = None,
) -> None:
    rows = []
    for i, (positive, negative) in enumerate(pairs):
        rows.append(
            {
                "pair_id": f"{kind}_{domain or 'all'}_{i:02d}",
                "kind": kind,
                "domain": domain,
                "positive": positive,
                "negative": negative,
                "positive_label": positive_label,
                "negative_label": negative_label,
                "mode": "prose",
                "matched_property": (
                    "semantic topic" if kind == "syntax" else "sentence form"
                ),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"wrote {path} n={len(rows)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_pairs(
        OUT / "syntax_control_pilot.jsonl",
        SYNTAX_PAIRS,
        kind="syntax",
        positive_label="imperative_or_declarative",
        negative_label="interrogative",
    )
    labels = {
        "coding": ("api", "algorithm"),
        "therapy": ("affect", "session_structure"),
        "writing": ("narrative", "expository"),
    }
    for domain, pairs in DOMAIN_PAIRS.items():
        pos, neg = labels[domain]
        write_pairs(
            OUT / f"domain_content_{domain}_pilot.jsonl",
            pairs,
            kind="domain_content",
            positive_label=pos,
            negative_label=neg,
            domain=domain,
        )
    # One preregistered engineering revision after the initial therapy control
    # failed its 0.6B split-half gate. Preserve v1; v2 expands n and uses more
    # tightly parallel sentence templates. Do not iterate again if v2 fails.
    write_pairs(
        OUT / "domain_content_therapy_pilot_v2.jsonl",
        [*DOMAIN_PAIRS["therapy"], *THERAPY_V2_ADDITIONS],
        kind="domain_content",
        positive_label="affect",
        negative_label="session_structure",
        domain="therapy",
    )
    # Writing v2: same one-shot expansion rule as therapy after L4 BCa lo grazed 0.70.
    write_pairs(
        OUT / "domain_content_writing_pilot_v2.jsonl",
        [*DOMAIN_PAIRS["writing"], *WRITING_V2_ADDITIONS],
        kind="domain_content",
        positive_label="narrative",
        negative_label="expository",
        domain="writing",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
