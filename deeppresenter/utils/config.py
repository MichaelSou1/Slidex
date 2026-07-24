import asyncio
import json
import random
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


class Endpoint(BaseModel):
    """LLM Endpoint Configuration"""

    base_url: str | None = Field(default=None, description="API base URL")
    model: str = Field(description="Model name")
    api_key: str | None = Field(default=None, description="API key")
    provider: Literal["openai", "litellm"] = Field(
        default="openai",
        description="Backend provider: 'openai' (default) or 'litellm'",
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Client parameters"
    )
    sampling_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Sampling parameters"
    )
    _client: AsyncOpenAI | None = PrivateAttr(default=None)

    def model_post_init(self, _) -> None:
        if self.provider != "litellm":
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.client_kwargs,
            )

    async def _call_litellm(
        self,
        messages: list[dict[str, Any]],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion:
        import litellm

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "drop_params": True,
            **self.sampling_parameters,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format is not None:
            kwargs["response_format"] = response_format
        return await litellm.acompletion(**kwargs)

    async def call(
        self,
        messages: list[dict[str, Any]],
        soft_response_parsing: bool,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion:
        """Execute a chat or tool call using the endpoint client"""
        if self.provider == "litellm":
            response = await self._call_litellm(messages, response_format, tools)
        elif tools is not None:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **self.sampling_parameters,
            )
        elif not soft_response_parsing and response_format is not None:
            response: ChatCompletion = await self._client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **self.sampling_parameters,
            )
        else:
            response: ChatCompletion = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.sampling_parameters,
            )
        assert response.choices is not None and len(response.choices) > 0, (
            f"No choices returned from the model, got {response}"
        )
        message = response.choices[0].message
        debug(f"Response from {self.model}: {message}")
        if response_format is not None:
            message.content = response_format(
                **get_json_from_response(message.content)
            ).model_dump_json(indent=2)
        assert tools is None or len(message.tool_calls or []), (
            f"No tool call returned from the model, got {message}"
        )
        assert message.tool_calls or message.content, (
            "Empty content returned from the model"
        )
        return response


class LLM(BaseModel):
    """LLM Client Manager"""

    base_url: str | None = Field(default=None, description="API base URL")
    model: str | None = Field(default=None, description="Model name")
    api_key: str | None = Field(default=None, description="API key")
    provider: str = Field(
        default="openai",
        description="Backend provider: 'openai' (default) or 'litellm'",
    )
    identifier: str | None = Field(
        default=None,
        description="Optional identifier for the model instance, this will override property `model_name`",
    )
    is_multimodal: bool | None = Field(
        default=None, description="Whether the model is multimodal"
    )
    max_concurrent: int | None = Field(
        default=None, description="Maximum concurrency limit"
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Client parameters"
    )
    sampling_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Sampling parameters"
    )
    endpoints: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Additional endpoints for alternating retries",
    )
    soft_response_parsing: bool = Field(
        default=False,
        description="Enable soft parsing: parse response content as JSON directly instead of using completion.parse",
    )
    min_image_size: int | None = Field(
        default=None,
        description="Minimum image size (width * height) for generation, smaller images will be resized proportionally",
    )
    secret_logging: bool = Field(
        default=False, description="Logging detailed endpoint (API key included)"
    )

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _endpoints: list[Endpoint] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def model_name(self) -> str:
        return self.identifier or self._endpoints[0].model.split("/")[-1].split(":")[0]

    def model_post_init(self, context) -> None:
        """Initialize semaphore and endpoints"""
        self._semaphore = asyncio.Semaphore(self.max_concurrent or 10000)
        if self.model:
            self._endpoints.insert(
                0,
                Endpoint(
                    base_url=self.base_url,
                    model=self.model,
                    api_key=self.api_key,
                    provider=self.provider,
                    client_kwargs=self.client_kwargs,
                    sampling_parameters=self.sampling_parameters,
                ),
            )
        for endpoint in self.endpoints:
            self._endpoints.append(Endpoint(**endpoint))
        assert len(self._endpoints) >= 1, "At least one endpoint must be configured"

        model_lower = self._endpoints[0].model.lower()
        if self.is_multimodal is None:
            if any(word in model_lower for word in ("gpt", "claude", "gemini", "vl")):
                self.is_multimodal = True
                debug(
                    f"Model {self._endpoints[0].model} is detected as multimodal model, setting `is_multimodal` to True"
                )
            else:
                self.is_multimodal = False

        return super().model_post_init(context)

    async def run(
        self,
        messages: list[dict[str, Any]] | str,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_times: int = RETRY_TIMES,
    ) -> ChatCompletion:
        """Unified interface for chat and tool calls with alternating retry"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        errors = []
        iter_endpoints = cycle(self._endpoints)
        async with self._semaphore:
            for _ in range(retry_times):
                endpoint = next(iter_endpoints)
                try:
                    return await endpoint.call(
                        messages,
                        self.soft_response_parsing,
                        response_format,
                        tools,
                    )
                except (AssertionError, ValidationError) as e:
                    errors.append(f"[{endpoint.model}] {e}")
                except Exception as e:
                    errors.append(f"[{endpoint.model}] {e}")
                    if self.secret_logging:
                        identifider = endpoint
                    else:
                        identifider = endpoint.model
                    logging_openai_exceptions(identifider, e)
        raise ValueError(f"All models failed after {retry_times} retries:\n{errors}")

    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        retry_times: int = RETRY_TIMES,
        pixel_multiple: int = PIXEL_MULTIPLE,
    ) -> ImagesResponse:
        """Unified interface for image generation"""
        if self.min_image_size is not None and (width * height) < int(
            self.min_image_size
        ):
            ratio = (int(self.min_image_size) / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        assert (width % pixel_multiple == 0) and (height % PIXEL_MULTIPLE == 0), (
            f"Image width and height must be a multiple of {pixel_multiple}"
        )
        async with self._semaphore:
            errors = []
            random.shuffle(self._endpoints)
            for retry_idx in range(retry_times):
                # t2i is stateless
                endpoint = self._endpoints[retry_idx % len(self._endpoints)]
                try:
                    if endpoint.provider == "litellm":
                        import litellm

                        response = await litellm.aimage_generation(
                            prompt=prompt,
                            model=endpoint.model,
                            size=f"{width}x{height}",
                            timeout=MCP_CALL_TIMEOUT // 5,
                            drop_params=True,
                            **(
                                {"api_key": endpoint.api_key}
                                if endpoint.api_key
                                else {}
                            ),
                            **(
                                {"api_base": endpoint.base_url}
                                if endpoint.base_url
                                else {}
                            ),
                            **endpoint.sampling_parameters,
                        )
                    else:
                        response = await endpoint._client.images.generate(
                            prompt=prompt,
                            model=endpoint.model,
                            size=f"{width}x{height}",
                            timeout=MCP_CALL_TIMEOUT // 5,
                            **endpoint.sampling_parameters,
                        )
                    assert len(response.data) >= 1, (
                        f"Expected at least an image response, got {response}"
                    )
                    return response

                except (AssertionError, ValidationError) as e:
                    errors.append(f"[{endpoint.model}] {e}")
                except Exception as e:
                    errors.append(f"[{endpoint.model}] {e}")
                    if self.secret_logging:
                        identifider = endpoint
                    else:
                        identifider = endpoint.model
                    logging_openai_exceptions(identifider, e)
            raise ValueError(f"All models failed after {retry_times} retries: {errors}")

    async def validate(self):
        endpoint = self._endpoints[0]
        if endpoint.provider == "litellm":
            import litellm

            try:
                await litellm.acompletion(
                    model=endpoint.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    drop_params=True,
                    **({"api_key": endpoint.api_key} if endpoint.api_key else {}),
                    **({"api_base": endpoint.base_url} if endpoint.base_url else {}),
                )
            except Exception as e:
                raise Exception(
                    f"LiteLLM validation failed for model {endpoint.model}: {e}\n"
                ) from e
            return
        models = await endpoint._client.models.list()
        if not any(model.id.endswith(endpoint.model) for model in models.data):
            raise Exception(
                f"Model {endpoint.model} is not available at {endpoint.base_url}, please check your apikey or {PACKAGE_DIR / 'config.yaml'}\n"
            )


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
        critic = value.get("critic_model") if isinstance(value, dict) else None
        if isinstance(critic, dict) and "is_multimodal" not in critic:
            raise ValueError("critic_model.is_multimodal must be explicitly declared")
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
