import asyncio
import json
import os
import tempfile
import threading
from enum import Enum
from functools import partial
from typing import Iterator, List, Optional, Type

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

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from google.oauth2 import service_account
from pydantic import ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Use langchain-core 0.2.x base classes directly to avoid the metaclass conflict
# introduced by langchain-google-vertexai 3.x (pydantic v2) vs BaseChatModel (pydantic v1).
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.embeddings import Embeddings

from cat.factory.llm import LLMSettings
from cat.factory.embedder import EmbedderSettings
from cat.log import log
from cat.mad_hatter.decorators import hook


class VertexAILocation(Enum):
    # Europe
    EU = "eu"
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
    """Write service-account credentials to a temp file and point ADC at it.

    google-genai picks up Application Default Credentials automatically from
    GOOGLE_APPLICATION_CREDENTIALS, so no vertexai.init() is needed.
    """
    info = json.loads(service_account_json)
    cred_path = os.path.join(tempfile.gettempdir(), "gcp_sa_credentials.json")
    with open(cred_path, "w") as f:
        json.dump(info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


# EU locations only: europe-west2 is London, outside the EU data residency boundary
_EU_FALLBACK_LOCATIONS = [
    "eu",
    "europe-west3",
    "europe-west1",
    "europe-west4",
    "europe-north1",
]


def _is_rate_limited(exc: BaseException) -> bool:
    # google-genai raises genai_errors.ClientError for all 4xx responses,
    # not google.api_core.exceptions.ResourceExhausted (that's the old
    # vertexai/gRPC SDK's exception, which this REST-based client never raises).
    return isinstance(exc, genai_errors.ClientError) and exc.code == 429


class VertexAIRateLimited(RuntimeError):
    """Raised when every configured EU region is still rate-limited after retries."""


# Quick exponential-backoff retry within a single region before the caller
# falls back to the next EU region on persistent rate limiting.
_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limited),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)


class GeminiVertexAIChat(BaseChatModel):
    """LLM backed by Google Vertex AI via the google-genai SDK.

    Inherits only from BaseChatModel (langchain-core 0.2.x, pydantic v1) so
    there is no metaclass conflict with the pydantic v2-based classes in
    langchain-google-vertexai 3.x.  All API calls go through google-genai
    which uses the modern Gemini REST endpoint — no deprecated Model Garden API.
    """

    project: str = ""
    location: str = "europe-west1"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192
    streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "gemini-vertexai"

    def _get_client(self) -> genai.Client:
        return genai.Client(vertexai=True, project=self.project, location=self.location)

    @staticmethod
    def _to_genai_contents(messages: List[BaseMessage]) -> List[genai_types.Content]:
        contents = []
        for msg in messages:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            role = "model" if isinstance(msg, AIMessage) else "user"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=text)])
            )
        return contents

    def _build_config(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> tuple:
        system_parts = []
        chat_msgs: List[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                system_parts.append(genai_types.Part.from_text(text=text))
            else:
                chat_msgs.append(msg)

        config_kw: dict = dict(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if system_parts:
            config_kw["system_instruction"] = genai_types.Content(parts=system_parts)
        if stop:
            config_kw["stop_sequences"] = stop
        return genai_types.GenerateContentConfig(**config_kw), chat_msgs

    @_retry_on_rate_limit
    def _call_generate_content(self, client, contents, config):
        return client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

    @_retry_on_rate_limit
    def _call_generate_content_stream(self, client, contents, config):
        return client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        client = self._get_client()
        config, chat_msgs = self._build_config(messages, stop)
        response = self._call_generate_content(
            client, self._to_genai_contents(chat_msgs), config
        )
        text = response.text or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        client = self._get_client()
        config, chat_msgs = self._build_config(messages, stop)
        for chunk in self._call_generate_content_stream(
            client, self._to_genai_contents(chat_msgs), config
        ):
            text = chunk.text or ""
            if text:
                if run_manager:
                    run_manager.on_llm_new_token(text)
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            partial(self._generate, messages, stop=stop, run_manager=run_manager, **kwargs),
        )


class GeminiVertexAIEmbeddings(Embeddings):
    """Embeddings backed by Google Vertex AI via the google-genai SDK."""

    def __init__(
        self, project: str, location: str, model_name: str = "gemini-embedding-001", **kwargs
    ):
        self.project = project
        self.location = location
        self.model_name = model_name

    def _get_client(self) -> genai.Client:
        return genai.Client(vertexai=True, project=self.project, location=self.location)

    @_retry_on_rate_limit
    def _call_embed_content(self, client, text):
        return client.models.embed_content(
            model=self.model_name,
            contents=text,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        result = []
        for text in texts:
            response = self._call_embed_content(client, text)
            result.append(list(response.embeddings[0].values))
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class RoundRobinChatVertexAI(GeminiVertexAIChat):
    """GeminiVertexAIChat that transparently retries with a different EU region on 429."""

    @classmethod
    def create(cls, sa_json: str, project: str, location: str, **kwargs) -> "RoundRobinChatVertexAI":
        locations = [location] + [l for l in _EU_FALLBACK_LOCATIONS if l != location]
        instance = cls(project=project, location=location, **kwargs)
        # BaseChatModel uses pydantic v1 which blocks __setattr__ for unknown fields;
        # store all mutable round-robin state in a single dict to bypass that guard.
        object.__setattr__(instance, "_rr_state", {
            "sa_json": sa_json,
            "project": project,
            "locations": locations,
            "index": 0,
            "lock": threading.Lock(),
            "kwargs": kwargs,
            "clients": {location: GeminiVertexAIChat(project=project, location=location, **kwargs)},
        })
        return instance

    def _get_or_create_client(self, location: str) -> GeminiVertexAIChat:
        state = self._rr_state
        if location not in state["clients"]:
            _init_vertex(state["sa_json"], state["project"], location)
            state["clients"][location] = GeminiVertexAIChat(
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
            except genai_errors.ClientError as e:
                if not _is_rate_limited(e):
                    raise
                log.warning(f"Vertex AI LLM rate-limited on {location}, switching EU region")
                location = self._next_location()
        raise VertexAIRateLimited("All configured Vertex AI EU regions are rate-limited")

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
            except genai_errors.ClientError as e:
                if not _is_rate_limited(e):
                    raise
                log.warning(f"Vertex AI LLM rate-limited on {location} (stream), switching EU region")
                location = self._next_location()
        raise VertexAIRateLimited("All configured Vertex AI EU regions are rate-limited")

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

    _pyclass: Type = GeminiVertexAIChat

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

    _pyclass: Type = GeminiVertexAIEmbeddings

    @classmethod
    def get_embedder_from_config(cls, config):
        project = config.pop("project_id").strip()
        location = config.pop("location").strip()
        _init_vertex(config.pop("service_account_json"), project, location)
        return GeminiVertexAIEmbeddings(project=project, location=location, **config)

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


# Expose a fake langchain_google_vertexai module so that code checking
# isinstance(cat._llm, ChatVertexAI) — e.g. audio_guardian.py — still works
# even though langchain-google-vertexai is not installed in this patched image.
import sys as _sys
import types as _types

if "langchain_google_vertexai" not in _sys.modules:
    _fake = _types.ModuleType("langchain_google_vertexai")
    _fake.ChatVertexAI = GeminiVertexAIChat
    _sys.modules["langchain_google_vertexai"] = _fake
