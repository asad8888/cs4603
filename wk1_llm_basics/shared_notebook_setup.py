"""Shared setup helpers for Databricks + OpenAI notebooks."""

from dataclasses import dataclass
import logging
import os

from dotenv import load_dotenv
from pprintpp import pprint
from langchain_openai import ChatOpenAI

import openai
import json
import warnings

# This dataclass holds the Databricks configuration loaded from environment variables.
# The `frozen=True` parameter makes it immutable, which is a good practice for configuration objects.
@dataclass(frozen=True)
class DatabricksConfig:
    token: str
    host: str
    endpoint: str


class MissingEnvironmentVariableError(ValueError):
    """Raised when one or more required environment variables are missing."""

# This function loads Databricks environment variables and returns a typed config object.
def get_databricks_config(validate: bool = True) -> DatabricksConfig:
    """Load Databricks environment variables and return a typed config object."""
    load_dotenv()

    token = os.environ.get("DATABRICKS_TOKEN", "")
    host = os.environ.get("DATABRICKS_HOST", "")
    model = os.environ.get("DATABRICKS_MODEL", "")

    # If validate is True, check for missing variables and raise an error if any are not set.
    if validate:
        missing = [
            name
            for name, value in {
                "DATABRICKS_TOKEN": token,
                "DATABRICKS_HOST": host,
                "DATABRICKS_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise MissingEnvironmentVariableError(
                f"Missing required environment variable(s): {missing_text}"
            )

    return DatabricksConfig(token=token, host=host, endpoint=model)


def create_databricks_client(config: DatabricksConfig) -> openai.OpenAI:
    """Create an OpenAI client configured for Databricks model serving endpoints."""
    return openai.OpenAI(
        api_key=config.token,
        base_url=f"{config.host}/serving-endpoints",
    )

def enable_logging():
    import logging

    logging.disable(logging.NOTSET)

    # root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    # for h in root.handlers:
    #     h.setLevel(logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG, force=True) # to show the communication embedding model and the vector store

def disable_logging():
    import logging

    logging.disable(logging.CRITICAL)
    
def bootstrap_notebook(validate: bool = True):
    """Return notebook-ready variables: token, host, endpoint, and configured client."""
    config = get_databricks_config(validate=validate)
    client = create_databricks_client(config)
    return config.token, config.host, config.endpoint, client

if __name__ == "__main__":
    warnings.filterwarnings("ignore", module="pydantic")
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
        warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
    except Exception:
        pass

    DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, client = bootstrap_notebook()
