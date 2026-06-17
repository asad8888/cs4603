"""Shared setup helpers for Databricks-native LangChain notebooks.

Uses the databricks-langchain package (ChatDatabricks, DatabricksEmbeddings)
and loads configuration from environment variables (DATABRICKS_TOKEN, DATABRICKS_HOST, etc)."""

from dataclasses import dataclass
import os
import warnings

from dotenv import load_dotenv
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

@dataclass(frozen=True)
class DatabricksConfig:
    token: str
    host: str
    endpoint: str
    vs_endpoint: str
    vs_index: str


@dataclass(frozen=True)
class VectorSearchConfig:
    endpoint_name: str
    index_name: str


class MissingEnvironmentVariableError(ValueError):
    """Raised when one or more required environment variables are missing."""


def get_databricks_config(validate: bool = True) -> DatabricksConfig:
    """Load Databricks environment variables and return a typed config object."""
    load_dotenv()

    token = os.environ.get("DATABRICKS_TOKEN", "")
    host = os.environ.get("DATABRICKS_HOST", "")
    model = os.environ.get("DATABRICKS_MODEL", "")
    vs_endpoint = os.environ.get("DATABRICKS_VS_ENDPOINT", "")
    vs_index = os.environ.get("DATABRICKS_VS_INDEX", "")

    if validate:
        missing = [
            name
            for name, value in {
                "DATABRICKS_TOKEN": token,
                "DATABRICKS_HOST": host,
                "DATABRICKS_MODEL": model,
                "DATABRICKS_VS_ENDPOINT": vs_endpoint,
                "DATABRICKS_VS_INDEX": vs_index,
            }.items()
            if not value
        ]
        if missing:
            raise MissingEnvironmentVariableError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )

    return DatabricksConfig(
        token=token,
        host=host,
        endpoint=model,
        vs_endpoint=vs_endpoint,
        vs_index=vs_index,
    )


def create_clients(config: DatabricksConfig):
    """Create ChatDatabricks LLM clients and DatabricksEmbeddings."""
    llm = ChatDatabricks(
        endpoint=config.endpoint,
        # DATABRICKS_HOST and DATABRICKS_TOKEN are auto-read from env by the SDK
    )
    # reasoning_effort="none" skips the thinking step — faster and cheaper
    llm_noreason = ChatDatabricks(
        endpoint=config.endpoint,
        extra_params={"reasoning_effort": "none"},
    )
    embeddings = DatabricksEmbeddings(
        endpoint="databricks-gte-large-en",
        # produces 1024-dimensional vectors
    )
    return llm, llm_noreason, embeddings


def enable_logging():
    import logging
    logging.disable(logging.NOTSET)
    logging.basicConfig(level=logging.DEBUG, force=True)


def disable_logging():
    import logging
    logging.disable(logging.CRITICAL)


def bootstrap_notebook(validate: bool = True):
    """Return notebook-ready variables for Databricks-native notebooks.

    Returns:
        token, host, endpoint, (llm, llm_noreason), embeddings, vs_config
    """
    warnings.filterwarnings("ignore", module="pydantic")
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
        warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
    except Exception:
        pass

    config = get_databricks_config(validate=validate)
    llm, llm_noreason, embeddings = create_clients(config)
    vs_config = VectorSearchConfig(
        endpoint_name=config.vs_endpoint,
        index_name=config.vs_index,
    )
    return config.token, config.host, config.endpoint, (llm, llm_noreason), embeddings, vs_config


if __name__ == "__main__":
    token, host, endpoint, (llm, llm_noreason), embeddings, vs_config = bootstrap_notebook()
    print(f"Connected to: {host}")
    print(f"Model endpoint: {endpoint}")
    print(f"Vector Search index: {vs_config.index_name}")
