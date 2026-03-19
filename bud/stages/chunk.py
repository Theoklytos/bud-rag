"""Chunking stage for Bud RAG Pipeline."""

import json
import uuid as uuid_lib

CHUNK_USER_TEMPLATE = """Conversation: {name}
Turns:
{turns_text}

Respond with JSON only:
{{
  "chunks": [
    {{
      "turns": [0, 1, 2],
      "tags": {{
        "geometry": "one of {geometry}",
        "coherence": "one of {coherence}",
        "texture": "one of {texture}",
        "terrain": "one of {terrain}",
        "motifs": ["one or more of {motifs}"]
      }},
      "chunk_type": "one of {chunk_types}",
      "split_rationale": "why cut here"
    }}
  ],
  "schema_proposals": [
    {{"dimension": "terrain", "value": "new_value", "rationale": "why"}}
  ]
}}"""


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, int(len(text.split()) * 1.3))


def _turns_to_text(turns: list[dict]) -> str:
    """Convert turns to display text for LLM prompt.

    Args:
        turns: List of turn dicts

    Returns:
        Formatted string with turn numbers and text
    """
    lines = []
    for i, t in enumerate(turns):
        text = t["text"][:600]
        lines.append(f"[{i}] {t['sender']}: {text}")
    return "\n".join(lines)


def _build_fallback_chunks(conversation: dict, prompt_preset: str) -> list[dict]:
    """Build fallback chunks when LLM fails.

    Args:
        conversation: Conversation dict
        prompt_preset: Prompt preset name

    Returns:
        List of fallback chunk dicts
    """
    chunks = []
    turns = conversation["turns"]
    for i in range(0, max(1, len(turns)), 2):
        pair = turns[i:i+2]
        text = " ".join(t["text"] for t in pair)
        chunks.append({
            "chunk_id": str(uuid_lib.uuid4()),
            "conversation_id": conversation["id"],
            "source_file": conversation["source_file"],
            "text": text,
            "turns": [i, i+1] if len(pair) > 1 else [i],
            "tags": {"geometry": "linear", "coherence": "loose",
                     "texture": "raw", "terrain": "conceptual", "motifs": []},
            "chunk_type": "exchange",
            "split_rationale": "fallback: LLM failure",
            "schema_version": 0,
            "llm_failure": True,
            "prompt_preset": prompt_preset,
            "schema_proposals": [],
        })
    return chunks


def chunk_conversation(
    conversation: dict,
    schema: dict,
    llm,
    config: dict,
    prompt: str,
    prompt_preset: str = "conversational",
    schema_version: int = 1,
    max_retries: int = 2,
) -> list[dict]:
    """Chunk a conversation using LLM.

    Args:
        conversation: Parsed conversation dict
        schema: Current schema dict
        llm: LLMClient instance
        config: Pipeline config
        prompt: System prompt
        prompt_preset: Name of the prompt preset
        schema_version: Current schema version
        max_retries: Number of retry attempts

    Returns:
        List of chunk dicts
    """
    dims = schema["dimensions"]
    user_msg = CHUNK_USER_TEMPLATE.format(
        name=conversation.get("conversation_name", "(unnamed)"),
        turns_text=_turns_to_text(conversation["turns"]),
        geometry=", ".join(dims["geometry"]),
        coherence=", ".join(dims["coherence"]),
        texture=", ".join(dims["texture"]),
        terrain=", ".join(dims["terrain"]),
        motifs=", ".join(dims["motifs"]),
        chunk_types=", ".join(schema["chunk_types"]),
    )

    min_tok = config["pipeline"]["chunk_min_tokens"]
    max_tok = config["pipeline"]["chunk_max_tokens"]

    data = None
    for attempt in range(max_retries + 1):
        try:
            response_text = llm.complete(system=prompt, user=user_msg)
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            break
        except Exception:
            if attempt == max_retries:
                return _build_fallback_chunks(conversation, prompt_preset)

    if data is None:
        return _build_fallback_chunks(conversation, prompt_preset)

    raw_chunks = data.get("chunks", [])
    proposals = data.get("schema_proposals", [])
    turns = conversation["turns"]
    chunks = []

    for i, rc in enumerate(raw_chunks):
        turn_indices = rc.get("turns", [])
        turn_texts = [turns[j]["text"] for j in turn_indices if j < len(turns)]
        text = " ".join(turn_texts)
        tok_count = estimate_tokens(text)

        if tok_count < min_tok and len(raw_chunks) > 1:
            continue
        chunk = {
            "chunk_id": str(uuid_lib.uuid4()),
            "conversation_id": conversation["id"],
            "source_file": conversation["source_file"],
            "text": text,
            "turns": turn_indices,
            "tags": rc.get("tags", {}),
            "chunk_type": rc.get("chunk_type", "exchange"),
            "split_rationale": rc.get("split_rationale", ""),
            "schema_version": schema_version,
            "llm_failure": False,
            "prompt_preset": prompt_preset,
            "schema_proposals": proposals if i == 0 else [],
        }
        chunks.append(chunk)

    return chunks if chunks else _build_fallback_chunks(conversation, prompt_preset)
