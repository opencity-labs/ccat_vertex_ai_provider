import asyncio
import json
import os
import tempfile
import threading
from enum import Enum
from functools import partial
from typing import List, Type

# google-auth 2.50.0 added CredentialsWithRegionalAccessBoundary to
# google.auth.credentials, but Docker layer caching can leave a stale
# compiled bytecode for that module while google/oauth2/credentials.py
# is already at 2.50.0 and references it at import time.  The patch below
# must run before any google.oauth2 import (including transitive ones from
# vertexai or langchain_google_vertexai) to prevent the AttributeError.
import google.auth.credentials as _gac
if not hasattr(_gac, "CredentialsWithRegionalAccessBoundary"):
    class _CredentialsWithRegionalAccessBoundary:
        pass
    _gac.CredentialsWithRegionalAccessBoundary = _CredentialsWithRegionalAccessBoundary

# langchain-google-vertexai>=3.x needs several symbols that only exist in
# langchain-core>=0.3, while the base image ships langchain-core~=0.2.
# Inject stubs for everything that chat_models.py v3.2.4 imports but that
# 0.2.x does not export, so the plugin loads cleanly at import time.
import sys
import types
import warnings

import langchain_core.language_models as _lcl
import langchain_core.language_models.base as _lcl_base
import langchain_core.messages as _lcm
import langchain_core.utils.function_calling as _lcuf
import langchain_core.utils.utils as _lcuu

def _patch(mod, name, stub):
    if not hasattr(mod, name):
        setattr(mod, name, stub)

# --- langchain_core.language_models ---
class _ModelProfile:
    pass
class _ModelProfileRegistry:
    pass
_patch(_lcl, "ModelProfile", _ModelProfile)
_patch(_lcl, "ModelProfileRegistry", _ModelProfileRegistry)
_patch(_lcl_base, "ModelProfile", _ModelProfile)
_patch(_lcl_base, "ModelProfileRegistry", _ModelProfileRegistry)

# --- langchain_core.messages: image/content helpers ---
def _convert_to_openai_image_block(image_data, **kwargs):
    if isinstance(image_data, dict):
        return image_data
    return {"type": "image_url", "image_url": {"url": str(image_data)}}

def _is_data_content_block(block):
    if not isinstance(block, dict):
        return False
    return block.get("type") in ("image_url", "image", "audio_url", "audio", "video_url", "file")

_patch(_lcm, "convert_to_openai_image_block", _convert_to_openai_image_block)
_patch(_lcm, "is_data_content_block", _is_data_content_block)

# --- langchain_core.messages.ai: InputTokenDetails ---
import langchain_core.messages.ai as _lcma
class _InputTokenDetails(dict):
    pass
_patch(_lcma, "InputTokenDetails", _InputTokenDetails)

# --- langchain_core.messages.content sub-module ---
if "langchain_core.messages.content" not in sys.modules:
    _content_mod = types.ModuleType("langchain_core.messages.content")
    sys.modules["langchain_core.messages.content"] = _content_mod
    _lcm.content = _content_mod

# --- langchain_core.utils.function_calling: convert_to_json_schema ---
def _convert_to_json_schema(schema, *, strict=None):
    try:
        kw = {} if strict is None else {"strict": strict}
        openai_tool = _lcuf.convert_to_openai_tool(schema, **kw)
        if isinstance(openai_tool, dict) and "function" in openai_tool:
            return openai_tool["function"].get("parameters", {})
    except Exception:
        pass
    return schema if isinstance(schema, dict) else {}

_patch(_lcuf, "convert_to_json_schema", _convert_to_json_schema)

# --- langchain_core.utils.utils: _build_model_kwargs ---
def _build_model_kwargs(values, all_required_field_names):
    extra_kwargs = values.get("model_kwargs", {})
    for field_name in list(values):
        if field_name in extra_kwargs:
            raise ValueError(f"Found {field_name} supplied twice.")
        if field_name not in all_required_field_names:
            warnings.warn(
                f"WARNING! {field_name} is not default parameter. "
                f"{field_name} was transferred to model_kwargs.",
                stacklevel=7,
            )
            extra_kwargs[field_name] = values.pop(field_name)
    values["model_kwargs"] = extra_kwargs
    return values

_patch(_lcuu, "_build_model_kwargs", _build_model_kwargs)

import vertexai
from google.oauth2 import service_account
from google.api_core.exceptions import ResourceExhausted
from pydantic import ConfigDict, Field
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from cat.factory.llm import LLMSettings
from cat.factory.embedder import EmbedderSettings
from cat.log import log
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


def _init_vertex(service_account_json: str, project: str, location: str):
    """Initialise the Vertex AI SDK with explicit credentials.

    langchain-google-vertexai==1.0.4 creates the embedder client via
    TextEmbeddingModel.from_pretrained(), which reads from the global vertexai
    SDK state rather than the credentials passed to VertexAIEmbeddings.
    Without an explicit vertexai.init() call the SDK falls back to ADC and
    fails when no ambient credentials are available.

    GOOGLE_APPLICATION_CREDENTIALS is also set for the gRPC transport layer,
    which does not receive the credentials object directly.
    """
    info = json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    vertexai.init(project=project, location=location, credentials=credentials)

    cred_path = os.path.join(tempfile.gettempdir(), "gcp_sa_credentials.json")
    with open(cred_path, "w") as f:
        json.dump(info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


# EU regions confirmed to support Gemini LLM and embedding models
_EU_FALLBACK_LOCATIONS = [
    "europe-west1",
    "europe-west2",
    "europe-west4",
    "europe-north1",
    "europe-west3",
]


class RoundRobinChatVertexAI(ChatVertexAI):
    """ChatVertexAI that transparently retries with a different EU region on 429."""

    @classmethod
    def create(cls, sa_json: str, project: str, location: str, **kwargs) -> "RoundRobinChatVertexAI":
        locations = [location] + [l for l in _EU_FALLBACK_LOCATIONS if l != location]
        instance = cls(project=project, location=location, **kwargs)
        # ChatVertexAI uses pydantic v1 which blocks __setattr__ for unknown fields;
        # store all mutable round-robin state in a single dict to bypass that guard.
        object.__setattr__(instance, "_rr_state", {
            "sa_json": sa_json,
            "project": project,
            "locations": locations,
            "index": 0,
            "lock": threading.Lock(),
            "kwargs": kwargs,
            # Plain ChatVertexAI (not self) for the primary region — avoids recursion in delegation
            "clients": {location: ChatVertexAI(project=project, location=location, **kwargs)},
        })
        return instance

    def _get_or_create_client(self, location: str) -> ChatVertexAI:
        state = self._rr_state
        if location not in state["clients"]:
            _init_vertex(state["sa_json"], state["project"], location)
            state["clients"][location] = ChatVertexAI(
                project=state["project"], location=location, **state["kwargs"]
            )
        return state["clients"][location]

    def _next_location(self) -> str:
        state = self._rr_state
        with state["lock"]:
            state["index"] = (state["index"] + 1) % len(state["locations"])
            return state["locations"][state["index"]]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        state = self._rr_state
        tried: set = set()
        location = state["locations"][state["index"]]
        for _ in range(len(state["locations"])):
            if location in tried:
                break
            tried.add(location)
            client = self._get_or_create_client(location)
            try:
                return client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except ResourceExhausted:
                log.warning(f"Vertex AI LLM rate-limited on {location}, switching EU region")
                location = self._next_location()
        raise ResourceExhausted("All configured Vertex AI EU regions are rate-limited")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        state = self._rr_state
        tried: set = set()
        location = state["locations"][state["index"]]
        for _ in range(len(state["locations"])):
            if location in tried:
                break
            tried.add(location)
            client = self._get_or_create_client(location)
            try:
                yield from client._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
                return
            except ResourceExhausted:
                log.warning(f"Vertex AI LLM rate-limited on {location} (stream), switching EU region")
                location = self._next_location()
        raise ResourceExhausted("All configured Vertex AI EU regions are rate-limited")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._generate, messages, stop=stop, run_manager=run_manager, **kwargs),
        )


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
        project = config.pop("project_id").strip()
        location = config.pop("location").strip()
        sa_json = config.pop("service_account_json")
        _init_vertex(sa_json, project, location)
        return RoundRobinChatVertexAI.create(sa_json=sa_json, project=project, location=location, **config)

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
        project = config.pop("project_id").strip()
        location = config.pop("location").strip()
        _init_vertex(config.pop("service_account_json"), project, location)
        return VertexAIEmbeddings(project=project, location=location, **config)

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
