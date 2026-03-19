"""Tests for error hierarchy in bud.lib.errors."""

from bud.lib.errors import (
    PipelineError,
    LLMError,
    LLMTimeoutError,
    LLMInvalidResponseError,
    EmbeddingError,
    SchemaError,
    StoreError,
)


def test_llm_timeout_is_llm_error():
    assert issubclass(LLMTimeoutError, LLMError)


def test_llm_error_is_pipeline_error():
    assert issubclass(LLMError, PipelineError)


def test_llm_invalid_response_is_llm_error():
    assert issubclass(LLMInvalidResponseError, LLMError)


def test_embedding_error_is_pipeline_error():
    assert issubclass(EmbeddingError, PipelineError)


def test_schema_error_is_pipeline_error():
    assert issubclass(SchemaError, PipelineError)


def test_store_error_is_pipeline_error():
    assert issubclass(StoreError, PipelineError)


def test_all_errors_are_exceptions():
    for cls in (PipelineError, LLMError, LLMTimeoutError, LLMInvalidResponseError,
                EmbeddingError, SchemaError, StoreError):
        assert issubclass(cls, Exception)
