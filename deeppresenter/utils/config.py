import asyncio
import hashlib
import json
import os
from itertools import cycle, product
from pathlib import Path
from typing import Any, Literal

import json_repair
import yaml
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.images_response import ImagesResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from deeppresenter.utils.constants import (
    CONTEXT_LENGTH_LIMIT,
    MCP_CALL_TIMEOUT,
    PACKAGE_DIR,
    PIXEL_MULTIPLE,
    RETRY_TIMES,
)
from deeppresenter.utils.log import debug, logging_openai_exceptions


def get_json_from_response(response: str) -> dict | list:
    """
    Extract JSON from a text response.

    Args:
        response (str): The response text.

    Returns:
        Dict|List: The extracted JSON.

    Raises:
        Exception: If JSON cannot be extracted from the response.
    """

    assert isinstance(response, str) and len(response) > 0, (
        "response must be a non-empty string"
    )
    response = response.strip()
    try:
        return json.loads(response)
    except Exception:
        pass

    # Try to find JSON by looking for matching braces
    open_braces = []
    close_braces = []

    for i, char in enumerate(response):
        if char == "{" or char == "[":
            open_braces.append(i)
        elif char == "}" or char == "]":
            close_braces.append(i)

    for i, j in product(open_braces, reversed(close_braces)):
        if i > j:
            continue
        try:
            json_obj = json.loads(response[i : j + 1])
            if isinstance(json_obj, (dict, list)):
                return max(
                    json_obj, json_repair.loads(response), key=lambda x: len(str(x))
                )
        except Exception:
            pass

    return json_repair.loads(response)


class ModelCapabilities(BaseModel):
    """Explicit Chat Completions features supported by an endpoint."""

    model_config = ConfigDict(extra="forbid")

    text: bool = True
    vision: bool = False
    tools: bool = False
    structured_output: bool = False


class ModelCapabilityError(RuntimeError):
    """Raised before a request that the configured endpoint cannot serve."""


class Endpoint(BaseModel):
    """One lazily connected outbound model endpoint."""

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = Field(
        default=None, description="OpenAI-compatible API base URL"
    )
    model: str = Field(description="Provider model name")
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    provider: Literal["openai", "litellm"] = "openai"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def resolve_api_key_environment(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        key = data.get("api_key")
        if isinstance(key, str) and (key.startswith("$") or key.startswith("env:")):
            name = key[1:] if key.startswith("$") else key[4:]
            if not name or name not in os.environ:
                raise ValueError(f"API key environment variable is not set: {name}")
            data["api_key"] = os.environ[name]
        return data

    def client(self) -> AsyncOpenAI:
        """Create the SDK client on first use, never during config loading."""
        if self.provider != "openai":
            raise RuntimeError("OpenAI SDK client requested for a LiteLLM endpoint")
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key or "not-set",
                base_url=self.base_url,
                **self.client_kwargs,
            )
        return self._client

    def require(
        self, capability: Literal["text", "vision", "tools", "structured_output"]
    ) -> None:
        if not getattr(self.capabilities, capability):
            raise ModelCapabilityError(
                f"Endpoint {self.model!r} does not declare {capability!r} capability"
            )

    async def _call_litellm(
        self,
        messages: list[dict[str, Any]],
        response_format: type[BaseModel] | dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> ChatCompletion:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "drop_params": False,
            **self.sampling_parameters,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await litellm.acompletion(**kwargs)

    async def call(
        self,
        messages: list[dict[str, Any]],
        soft_response_parsing: bool,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatCompletion:
        """Execute the supported Chat Completions subset."""
        self.require("text")
        if any(
            block.get("type") == "image_url"
            for message in messages
            for block in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
        ):
            self.require("vision")
        if tools is not None:
            self.require("tools")
        if response_format is not None:
            self.require("structured_output")

        if self.provider == "litellm":
            response = await self._call_litellm(
                messages, response_format, tools, tool_choice
            )
        elif (
            tools is not None
            or soft_response_parsing
            or response_format is None
            or isinstance(response_format, dict)
        ):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                **self.sampling_parameters,
            }
            if tools is not None:
                kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
            if isinstance(response_format, dict):
                kwargs["response_format"] = response_format
            response = await self.client().chat.completions.create(**kwargs)
        else:
            response = await self.client().chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **self.sampling_parameters,
            )

        if not response.choices:
            raise ValueError(f"No choices returned from model {self.model}")
        message = response.choices[0].message
        if response_format is not None and isinstance(response_format, type):
            if not message.content:
                raise ValueError("Structured response has no content")
            message.content = response_format(
                **get_json_from_response(message.content)
            ).model_dump_json(indent=2)
        if tools is not None and not message.tool_calls:
            raise ValueError("Provider returned no tool calls for a tool request")
        for tool_call in message.tool_calls or []:
            function = tool_call.function
            if not function.name or function.arguments is None:
                raise ValueError("Provider returned an incomplete tool call")
            try:
                arguments = json.loads(function.arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Provider returned invalid tool-call arguments"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
        if not message.tool_calls and not message.content:
            raise ValueError("Provider returned empty assistant content")
        debug(f"Response from {self.model}: {message}")
        return response


class LLM(BaseModel):
    """Unified outbound client with endpoint rotation and auditable calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    provider: Literal["openai", "litellm"] = "openai"
    identifier: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    is_multimodal: bool | None = Field(default=None, exclude=True)
    max_concurrent: int | None = None
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[dict[str, Any]] = Field(
        default_factory=list, exclude=True, repr=False
    )
    soft_response_parsing: bool = False
    min_image_size: int | None = None
    secret_logging: bool = False

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _endpoints: list[Endpoint] = PrivateAttr(default_factory=list)
    _last_call: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        capabilities = dict(data.get("capabilities") or {})
        if "is_multimodal" in data:
            capabilities.setdefault("vision", bool(data["is_multimodal"]))
            capabilities.setdefault("structured_output", True)
        data["capabilities"] = capabilities
        key = data.get("api_key")
        if isinstance(key, str) and (key.startswith("$") or key.startswith("env:")):
            name = key[1:] if key.startswith("$") else key[4:]
            if not name or name not in os.environ:
                raise ValueError(f"API key environment variable is not set: {name}")
            data["api_key"] = os.environ[name]
        return data

    @property
    def model_name(self) -> str:
        return self.identifier or self._endpoints[0].model.split("/")[-1].split(":")[0]

    @property
    def last_call(self) -> dict[str, Any] | None:
        """Return a copy of provenance for the most recent completed request."""
        return dict(self._last_call) if self._last_call else None

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent or 10000)
        if self.model:
            self._endpoints.append(
                Endpoint(
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    provider=self.provider,
                    capabilities=self.capabilities,
                    client_kwargs=self.client_kwargs,
                    sampling_parameters=self.sampling_parameters,
                )
            )
        for endpoint_data in self.endpoints:
            endpoint = dict(endpoint_data)
            endpoint.setdefault("provider", self.provider)
            endpoint.setdefault("capabilities", self.capabilities.model_dump())
            self._endpoints.append(Endpoint(**endpoint))
        if not self._endpoints:
            raise ValueError("At least one endpoint must be configured")
        self.is_multimodal = self._endpoints[0].capabilities.vision
        super().model_post_init(context)

    def require_capabilities(
        self, *capabilities: Literal["text", "vision", "tools", "structured_output"]
    ) -> None:
        """Fail early unless every configured rotation endpoint supports each feature."""
        for endpoint in self._endpoints:
            for capability in capabilities:
                endpoint.require(capability)

    async def run(
        self,
        messages: list[dict[str, Any]] | str,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_times: int = RETRY_TIMES,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatCompletion:
        """Run one logical request, rotating endpoints only when retries are enabled."""
        if retry_times < 1:
            raise ValueError("retry_times must be at least one")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        errors: list[str] = []
        iter_endpoints = cycle(self._endpoints)
        async with self._semaphore:
            for attempt in range(1, retry_times + 1):
                endpoint = next(iter_endpoints)
                try:
                    response = await endpoint.call(
                        messages,
                        self.soft_response_parsing,
                        response_format,
                        tools,
                        tool_choice,
                    )
                    payload = response.model_dump_json(exclude_none=True)
                    self._last_call = {
                        "endpoint_identifier": self.identifier
                        or endpoint.base_url
                        or "openai-default",
                        "provider": endpoint.provider,
                        "model": endpoint.model,
                        "sampling_parameters": dict(endpoint.sampling_parameters),
                        "usage": response.usage.model_dump(mode="json")
                        if response.usage
                        else {},
                        "finish_reasons": [
                            choice.finish_reason for choice in response.choices
                        ],
                        "reasoning": [
                            getattr(choice.message, "reasoning", None)
                            for choice in response.choices
                        ],
                        "tool_calls": [
                            [
                                call.model_dump(mode="json")
                                for call in (choice.message.tool_calls or [])
                            ]
                            for choice in response.choices
                        ],
                        "response_hash": hashlib.sha256(payload.encode()).hexdigest(),
                        "attempt": attempt,
                    }
                    return response
                except (ModelCapabilityError, ValidationError):
                    raise
                except Exception as exc:
                    errors.append(f"[{endpoint.model}] {type(exc).__name__}: {exc}")
                    logging_openai_exceptions(
                        endpoint if self.secret_logging else endpoint.model, exc
                    )
        raise ValueError(f"All models failed after {retry_times} retries:\n{errors}")

    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        retry_times: int = RETRY_TIMES,
        pixel_multiple: int = PIXEL_MULTIPLE,
    ) -> ImagesResponse:
        """Generate an image through the configured provider adapter."""
        if self.min_image_size is not None and width * height < self.min_image_size:
            ratio = (self.min_image_size / (width * height)) ** 0.5
            width, height = int(width * ratio), int(height * ratio)
        if width % pixel_multiple or height % pixel_multiple:
            raise ValueError(f"Image dimensions must be multiples of {pixel_multiple}")
        errors: list[str] = []
        async with self._semaphore:
            for retry_idx in range(retry_times):
                endpoint = self._endpoints[retry_idx % len(self._endpoints)]
                try:
                    if endpoint.provider == "litellm":
                        import litellm

                        return await litellm.aimage_generation(
                            prompt=prompt,
                            model=endpoint.model,
                            size=f"{width}x{height}",
                            timeout=MCP_CALL_TIMEOUT // 5,
                            api_key=endpoint.api_key,
                            api_base=endpoint.base_url,
                            **endpoint.sampling_parameters,
                        )
                    return await endpoint.client().images.generate(
                        prompt=prompt,
                        model=endpoint.model,
                        size=f"{width}x{height}",
                        timeout=MCP_CALL_TIMEOUT // 5,
                        **endpoint.sampling_parameters,
                    )
                except Exception as exc:
                    errors.append(f"[{endpoint.model}] {type(exc).__name__}: {exc}")
        raise ValueError(
            f"All image models failed after {retry_times} retries: {errors}"
        )

    async def validate(self) -> None:
        """Explicitly connect and verify the Chat Completions subset with a tiny request."""
        await self.run("ping", retry_times=1)


class SlidexConfig(BaseModel):
    """Configuration for versioned Slidex inspection and persistence."""

    model_config = ConfigDict(extra="forbid")

    taxonomy_version: str = "1.0"
    router_version: str = "1.0"
    reward_version: str = "1.0"
    reward_terminal_hard_negative: float = Field(default=-1, ge=-1, le=0)
    reward_severe_defect_threshold: float = Field(default=0.7, ge=0, le=1)
    reward_inspector_error_penalty: float = Field(default=0.2, ge=0, le=1)
    reward_inspector_error_invalidation_count: int = Field(default=2, ge=1)
    reward_policy_penalty_per_severity: float = Field(default=0.25, ge=0, le=1)
    safety_margin_px: float = Field(default=24, ge=0)
    alignment_tolerance_px: float = Field(default=2, ge=0)
    overlap_tolerance_px: float = Field(default=1, ge=0)
    palette_threshold: float = Field(default=0.1, ge=0, le=1)
    overlap_min_area_px: float = Field(default=4, ge=0)
    color_delta_e_threshold: float = Field(default=5, ge=0)
    typography_tolerance_px: float = Field(default=1, ge=0)
    max_repair_rounds: int = Field(default=3, ge=0)
    max_episode_steps: int = Field(default=20, gt=0)
    command_timeout_seconds: int = Field(default=300, gt=0)
    strict_export: bool = True
    pptx_rerender: bool = True
    export_max_pixel_difference: float = Field(default=0.12, ge=0, le=1)
    export_min_perceptual_similarity: float = Field(default=0.90, ge=0, le=1)
    export_min_text_presence: float = Field(default=0.95, ge=0, le=1)
    mutation_zero_signal_threshold: float = Field(default=0.001, ge=0, le=1)
    reference_policy: Literal["never", "on_defer", "always"] = "on_defer"
    max_workspace_bytes: int = Field(default=2 * 1024**3, gt=0)
    max_artifacts_per_episode: int = Field(default=1000, gt=0)
    artifact_retention_seconds: int = Field(default=7 * 24 * 3600, ge=0)


class DeepPresenterConfig(BaseModel):
    """DeepPresenter Global Configuration"""

    model_config = ConfigDict(extra="forbid")

    # config
    multiagent_mode: bool = Field(
        default=False, description="Enable multiagent mode (experimental)"
    )
    offline_mode: bool = Field(
        default=False, description="Enable offline mode, disable all network requests"
    )
    async_tool_mode: bool = Field(
        default=False,
        description="Enable async tool mode for slow tool calls",
    )
    file_path: str = Field(description="Configuration file path")
    mcp_config_file: str = Field(
        description="MCP configuration file", default=str(PACKAGE_DIR / "mcp.json")
    )
    context_folding: bool = Field(
        default=True, description="Enable context management and auto summarization"
    )
    context_window: int | None = Field(
        default=None,
        description="Context window for context management, if not set, use the default value",
    )
    max_context_folds: int = Field(
        default=5, description="Maximum number of folds for context management"
    )
    heavy_reflect: bool = Field(
        default=False,
        description="Enable heavy reflection, use rendered slide image for reflective design",
    )

    # llms
    research_agent: LLM = Field(description="Research agent model configuration")
    design_agent: LLM = Field(description="Design agent model configuration")
    long_context_model: LLM = Field(description="Long context model configuration")
    vision_model: LLM | None = Field(
        default=None, description="Vision model configuration"
    )
    t2i_model: LLM | None = Field(
        default=None, description="Text-to-image model configuration"
    )
    critic_model: LLM | None = Field(
        default=None, description="Independent critic model configuration"
    )
    semantic_model: LLM | None = Field(
        default=None, description="Optional independent semantic critic model"
    )
    slidex: SlidexConfig = Field(default_factory=SlidexConfig)

    @model_validator(mode="before")
    @classmethod
    def validate_critic_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        critic = value.get("critic_model")
        if isinstance(critic, dict):
            capabilities = critic.get("capabilities")
            legacy_vision = critic.get("is_multimodal")
            if capabilities is None and legacy_vision is None:
                raise ValueError(
                    "critic_model capabilities must be explicitly declared"
                )
            if isinstance(capabilities, dict):
                missing = {"vision", "structured_output"} - capabilities.keys()
                if missing:
                    raise ValueError(
                        "critic_model must explicitly declare capabilities: "
                        + ", ".join(sorted(missing))
                    )
        return value

    def model_post_init(self, context):
        if self.context_window is None:
            if self.context_folding:
                self.context_window = CONTEXT_LENGTH_LIMIT // self.max_context_folds
            else:
                self.context_window = CONTEXT_LENGTH_LIMIT

        if self.context_folding:
            debug(
                f"Context folding is enabled, context window: {self.context_window}, max folds: {self.max_context_folds}"
            )
        else:
            debug(f"Context folding is disabled, context window: {self.context_window}")

        return super().model_post_init(context)

    @classmethod
    def load_from_file(cls, config_path: str | None = None) -> "DeepPresenterConfig":
        """Load configuration from file"""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = PACKAGE_DIR / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist")
        config_data = {}
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        config_data["file_path"] = str(config_file.resolve())
        return cls(**config_data)

    async def validate_llms(self):
        # ? t2i endpoints might not support this api
        tasks = [
            self.research_agent.validate(),
            self.design_agent.validate(),
            self.long_context_model.validate(),
        ]
        if self.vision_model is not None:
            tasks.append(self.vision_model.validate())
        await asyncio.gather(*tasks)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
