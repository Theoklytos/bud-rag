"""Tests for bud.stages.chunk."""

import json
from unittest.mock import MagicMock

import pytest

from bud.lib.errors import LLMError
from bud.stages.chunk import chunk_conversation, _build_fallback_chunks, estimate_tokens


SAMPLE_SCHEMA = {
    "version": 1,
    "dimensions": {
        "geometry": ["linear", "recursive"],
        "coherence": ["tight", "loose"],
        "texture": ["dense", "raw"],
        "terrain": ["conceptual", "procedural"],
        "motifs": ["identity", "threshold"],
    },
    "chunk_types": ["exchange", "monologue"],
    "multi_value_dimensions": ["motifs"],
    "candidates": {},
    "evolution_log": [],
}

SAMPLE_CONVERSATION = {
    "id": "conv-1",
    "source_file": "conversations_001.json",
    "conversation_name": "Test Chat",
    "turns": [
        {"sender": "human", "text": "Hello"},
        {"sender": "assistant", "text": "Hi there, how can I help?"},
    ],
}

PIPELINE_CONFIG = {
    "pipeline": {
        "chunk_min_tokens": 1,
        "chunk_max_tokens": 2000,
    }
}

VALID_LLM_RESPONSE = json.dumps({
    "chunks": [
        {
            "turns": [0, 1],
            "tags": {
                "geometry": "linear",
                "coherence": "tight",
                "texture": "raw",
                "terrain": "conceptual",
                "motifs": ["identity"],
            },
            "chunk_type": "exchange",
            "split_rationale": "single exchange",
        }
    ],
    "schema_proposals": [],
})


def test_chunk_conversation_returns_chunks():
    llm = MagicMock()
    llm.complete.return_value = VALID_LLM_RESPONSE
    chunks = chunk_conversation(
        SAMPLE_CONVERSATION, SAMPLE_SCHEMA, llm, PIPELINE_CONFIG,
        prompt="system prompt", prompt_preset="conversational",
    )
    assert len(chunks) >= 1
    assert chunks[0]["conversation_id"] == "conv-1"
    assert "chunk_id" in chunks[0]


def test_chunk_conversation_calls_llm_complete():
    llm = MagicMock()
    llm.complete.return_value = VALID_LLM_RESPONSE
    chunk_conversation(
        SAMPLE_CONVERSATION, SAMPLE_SCHEMA, llm, PIPELINE_CONFIG,
        prompt="system prompt",
    )
    llm.complete.assert_called_once()


def test_fallback_chunks_when_llm_fails():
    llm = MagicMock()
    llm.complete.side_effect = LLMError("api down")
    chunks = chunk_conversation(
        SAMPLE_CONVERSATION, SAMPLE_SCHEMA, llm, PIPELINE_CONFIG,
        prompt="system prompt", max_retries=0,
    )
    assert len(chunks) >= 1
    assert chunks[0]["llm_failure"] is True


def test_fallback_chunks_when_llm_returns_invalid_json():
    llm = MagicMock()
    llm.complete.return_value = "not valid json at all"
    chunks = chunk_conversation(
        SAMPLE_CONVERSATION, SAMPLE_SCHEMA, llm, PIPELINE_CONFIG,
        prompt="system prompt", max_retries=0,
    )
    assert len(chunks) >= 1
    assert chunks[0]["llm_failure"] is True


def test_build_fallback_chunks_includes_all_fields():
    chunks = _build_fallback_chunks(SAMPLE_CONVERSATION, "conversational")
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert c["conversation_id"] == "conv-1"
        assert c["llm_failure"] is True
        assert c["prompt_preset"] == "conversational"


def test_estimate_tokens_positive():
    assert estimate_tokens("hello world foo bar") > 0


def test_estimate_tokens_single_word():
    assert estimate_tokens("hello") >= 1


def test_chunk_records_prompt_preset():
    llm = MagicMock()
    llm.complete.return_value = VALID_LLM_RESPONSE
    chunks = chunk_conversation(
        SAMPLE_CONVERSATION, SAMPLE_SCHEMA, llm, PIPELINE_CONFIG,
        prompt="sys", prompt_preset="mythic",
    )
    assert chunks[0]["prompt_preset"] == "mythic"
