"""Parsing stage for Bud RAG Pipeline."""

import glob
import json
import os
from pathlib import Path


def _load_memory_context(data_dir: Path) -> str | None:
    """Load memory context from memory files."""
    paths = list(data_dir.glob("memories_*.json"))
    for path in sorted(paths):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data[0].get("conversations_memory")
    return None


def _extract_blocks(content: list, fallback_text: str) -> tuple[str, str, bool, int]:
    """Extract text and thinking blocks from message content.

    Args:
        content: Message content list
        fallback_text: Fallback text if no text block

    Returns:
        Tuple of (text, thinking, thinking_truncated, dropped_blocks)
    """
    texts, thinkings = [], []
    thinking_truncated = False
    dropped = 0
    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "thinking":
            thinkings.append(block.get("thinking", ""))
            if block.get("cut_off") or block.get("truncated"):
                thinking_truncated = True
        elif btype in ("tool_use", "tool_result", "token_budget"):
            dropped += 1
    text = " ".join(t for t in texts if t).strip() or fallback_text
    thinking = "\n\n".join(t for t in thinkings if t)
    return text, thinking, thinking_truncated, dropped


def parse_conversations_file(path: Path, memory_context: str | None) -> list[dict]:
    """Parse a single conversations JSON file.

    Args:
        path: Path to the JSON file
        memory_context: Optional memory context string

    Returns:
        List of parsed conversation dicts
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        return []
    results = []
    for conv in raw:
        messages = conv.get("chat_messages", [])
        if not messages:
            continue
        turns = []
        for msg in messages:
            content = msg.get("content") or []
            text, thinking, thinking_truncated, dropped = _extract_blocks(
                content, msg.get("text", "")
            )
            turns.append({
                "turn_id": msg["uuid"],
                "sender": msg["sender"],
                "text": text,
                "thinking": thinking,
                "thinking_truncated": thinking_truncated,
                "dropped_blocks": dropped,
                "created_at": msg.get("created_at", ""),
            })
        results.append({
            "id": conv["uuid"],
            "source_file": path.name,
            "conversation_name": conv.get("name", ""),
            "conversation_summary": conv.get("summary", ""),
            "created_at": conv.get("created_at", ""),
            "turns": turns,
            "memory_context": memory_context,
        })
    return results


def parse_conversations_file_with_progress(
    path: Path, memory_context: str | None, progress
) -> list[dict]:
    """Parse a conversations file with progress callback.

    Args:
        path: Path to the JSON file
        memory_context: Optional memory context string
        progress: Progress callback function

    Returns:
        List of parsed conversation dicts
    """
    conversations = parse_conversations_file(path, memory_context)
    progress(len(conversations))
    return conversations


def parse_all(data_dir: Path, output_dir: Path, progress_callback=None) -> int:
    """Parse all conversations files in data_dir.

    Args:
        data_dir: Directory containing conversations_*.json files
        output_dir: Output directory for parsed files
        progress_callback: Optional callback(conversations_count)

    Returns:
        Total number of conversations parsed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_context = _load_memory_context(data_dir)
    total = 0

    for path in sorted(data_dir.glob("conversations_*.json")):
        conversations = parse_conversations_file(path, memory_context)

        out_path = output_dir / path.name.replace(".json", ".jsonl")
        with open(out_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        total += len(conversations)
        if progress_callback:
            progress_callback(total)

    return total
