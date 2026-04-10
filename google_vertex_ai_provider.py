import json
import os
import tempfile
from enum import Enum
from typing import List, Type

from pydantic import ConfigDict, Field
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from cat.factory.llm import LLMSettings
from cat.factory.embedder import EmbedderSettings
from cat.mad_hatter.decorators import hook


class VertexAILocation(Enum):
    # Europe
    EUROPE_WEST1 = "europe-west1"
    EUROPE_WEST2 = "europe-west2"
    EUROPE_WEST3 = "europe-west3"
    EUROPE_WEST4 = "europe-west4"
    EUROPE_WEST6 = "europe-west6"
    EUROPE_WEST8 = "europe-west8"
    EUROPE_WEST9 = "europe-west9"
    EUROPE_WEST12 = "europe-west12"
    EUROPE_NORTH1 = "europe-north1"
    EUROPE_CENTRAL2 = "europe-central2"
    EUROPE_SOUTHWEST1 = "europe-southwest1"
    # United States
    US_CENTRAL1 = "us-central1"
    US_EAST1 = "us-east1"
    US_EAST4 = "us-east4"
    US_EAST5 = "us-east5"
    US_SOUTH1 = "us-south1"
    US_WEST1 = "us-west1"
    US_WEST2 = "us-west2"
    US_WEST3 = "us-west3"
    US_WEST4 = "us-west4"
    # Canada
    NORTHAMERICA_NORTHEAST1 = "northamerica-northeast1"
    NORTHAMERICA_NORTHEAST2 = "northamerica-northeast2"
    # South America
    SOUTHAMERICA_EAST1 = "southamerica-east1"
    SOUTHAMERICA_WEST1 = "southamerica-west1"
    # Africa
    AFRICA_SOUTH1 = "africa-south1"
    # Asia Pacific
    ASIA_EAST1 = "asia-east1"
    ASIA_EAST2 = "asia-east2"
    ASIA_NORTHEAST1 = "asia-northeast1"
    ASIA_NORTHEAST2 = "asia-northeast2"
    ASIA_NORTHEAST3 = "asia-northeast3"
    ASIA_SOUTH1 = "asia-south1"
    ASIA_SOUTH2 = "asia-south2"
    ASIA_SOUTHEAST1 = "asia-southeast1"
    ASIA_SOUTHEAST2 = "asia-southeast2"
    AUSTRALIA_SOUTHEAST1 = "australia-southeast1"
    AUSTRALIA_SOUTHEAST2 = "australia-southeast2"
    # Middle East
    ME_CENTRAL1 = "me-central1"
    ME_CENTRAL2 = "me-central2"
    ME_WEST1 = "me-west1"
    # Global
    GLOBAL = "global"


def _setup_credentials(service_account_json: str):
    """Write service account JSON to a temp file and set GOOGLE_APPLICATION_CREDENTIALS.

    This is needed because langchain-google-vertexai's internal gRPC client
    does not forward the credentials object, falling back to ADC instead.
    """
    info = json.loads(service_account_json)
    cred_path = os.path.join(tempfile.gettempdir(), "gcp_sa_credentials.json")
    with open(cred_path, "w") as f:
        json.dump(info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


# --- LLM Provider ---


class LLMGoogleVertexAIConfig(LLMSettings):
    project_id: str
    location: VertexAILocation = VertexAILocation.EUROPE_WEST1
    service_account_json: str = Field(default="", extra={"type": "TextArea"})
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192
    streaming: bool = True

    _pyclass: Type = ChatVertexAI

    @classmethod
    def get_llm_from_config(cls, config):
        _setup_credentials(config.pop("service_account_json"))
        return ChatVertexAI(
            project=config.pop("project_id").strip(),
            location=config.pop("location").strip(),
            **config,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Google Vertex AI",
            "description": "LLM provider using Google Vertex AI with service account authentication.",
            "link": "https://cloud.google.com/vertex-ai",
        }
    )


# --- Embedder Provider ---


class EmbedderGoogleVertexAIConfig(EmbedderSettings):
    project_id: str
    location: VertexAILocation = VertexAILocation.EUROPE_WEST1
    service_account_json: str = Field(default="", extra={"type": "TextArea"})
    model_name: str = "gemini-embedding-001"

    _pyclass: Type = VertexAIEmbeddings

    @classmethod
    def get_embedder_from_config(cls, config):
        _setup_credentials(config.pop("service_account_json"))
        return VertexAIEmbeddings(
            project=config.pop("project_id").strip(),
            location=config.pop("location").strip(),
            **config,
        )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Google Vertex AI Embedder",
            "description": "Embedder using Google Vertex AI with service account authentication.",
            "link": "https://cloud.google.com/vertex-ai",
        }
    )


# --- Hooks to register providers ---


@hook
def factory_allowed_llms(allowed: List, cat) -> List:
    allowed.append(LLMGoogleVertexAIConfig)
    return allowed


@hook
def factory_allowed_embedders(allowed: List, cat) -> List:
    allowed.append(EmbedderGoogleVertexAIConfig)
    return allowed
