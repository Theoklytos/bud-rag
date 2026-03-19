"""Error types for Bud RAG Pipeline."""


class PipelineError(Exception):
    """Base error for pipeline issues."""


class LLMError(PipelineError):
    """Error from LLM API calls."""


class LLMTimeoutError(LLMError):
    """LLM request timed out."""


class LLMInvalidResponseError(LLMError):
    """LLM returned invalid response."""


class EmbeddingError(PipelineError):
    """Error from embedding API calls."""


class SchemaError(PipelineError):
    """Error from schema operations."""


class StoreError(PipelineError):
    """Error from vector store operations."""
