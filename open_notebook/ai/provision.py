import asyncio
import time
from typing import Any, Optional, Tuple

from esperanto import LanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from open_notebook.ai.models import model_manager
from open_notebook.exceptions import ConfigurationError
from open_notebook.utils import token_count


# --------------------------------------------------------------------------- #
# In-process cache for provisioned LangChain models.                          #
#                                                                             #
# `provision_langchain_model()` is called on every chat/transformation turn   #
# and does: DB lookup for model record, DB lookup for linked credential,      #
# Fernet decryption, Esperanto factory instantiation, and LangChain wrapping. #
# That is 100–300ms of pure overhead per turn.  These results are deterministic#
# given (model_id, purpose, size_bucket, kwargs) so we cache the provisioned  #
# BaseChatModel and reuse it for subsequent turns.                            #
#                                                                             #
# Invalidation:                                                               #
#   * `invalidate_model_cache()` — clears all entries (call on credential or  #
#     model updates via `POST/PUT /credentials`, `PUT /models/defaults`).     #
#   * TTL safety net: entries older than _MODEL_CACHE_TTL_SECONDS are         #
#     discarded to pick up out-of-band changes (e.g. direct DB edits).        #
# --------------------------------------------------------------------------- #

_MODEL_CACHE_TTL_SECONDS = 600  # 10 minutes
_model_cache: dict[Tuple[Any, ...], Tuple[float, BaseChatModel]] = {}
_model_cache_lock = asyncio.Lock()


def _cache_key(
    model_id: Optional[str],
    default_type: str,
    size_bucket: str,
    kwargs: dict,
) -> Tuple[Any, ...]:
    # Sort kwargs for stable keys; skip unhashable values by stringifying
    items = tuple(
        (k, v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
        for k, v in sorted(kwargs.items())
    )
    return (model_id, default_type, size_bucket, items)


def invalidate_model_cache() -> None:
    """Drop all cached provisioned models.

    Call after mutations to Model records, Credential records, or
    DefaultModels — i.e. anywhere a future `provision_langchain_model()` call
    could legitimately need to resolve to a different backing client.
    """
    _model_cache.clear()
    logger.debug("Provisioned model cache invalidated")


async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """
    Returns the best model to use based on the context size and on whether there is a specific model being requested in Config.
    If context > 105_000, returns the large_context_model
    If model_id is specified in Config, returns that model
    Otherwise, returns the default model for the given type
    """
    tokens = token_count(content)
    # Size bucket is part of the cache key so a "short" turn and a later
    # "large" turn on the same session don't collide and get the wrong model.
    size_bucket = "large" if tokens > 105_000 else "normal"

    key = _cache_key(model_id, default_type, size_bucket, kwargs)

    # Fast path: hit
    cached = _model_cache.get(key)
    if cached is not None:
        cached_at, cached_model = cached
        if time.monotonic() - cached_at <= _MODEL_CACHE_TTL_SECONDS:
            return cached_model
        # Expired — fall through to re-provision
        _model_cache.pop(key, None)

    # Slow path: provision + populate cache under lock (avoid thundering herd)
    async with _model_cache_lock:
        # Re-check inside the lock in case another coroutine just populated it
        cached = _model_cache.get(key)
        if cached is not None:
            cached_at, cached_model = cached
            if time.monotonic() - cached_at <= _MODEL_CACHE_TTL_SECONDS:
                return cached_model

        selection_reason = ""
        model = None

        if size_bucket == "large":
            selection_reason = f"large_context (content has {tokens} tokens)"
            logger.debug(
                f"Using large context model because the content has {tokens} tokens"
            )
            model = await model_manager.get_default_model("large_context", **kwargs)
        elif model_id:
            selection_reason = f"explicit model_id={model_id}"
            model = await model_manager.get_model(model_id, **kwargs)
        else:
            selection_reason = f"default for type={default_type}"
            model = await model_manager.get_default_model(default_type, **kwargs)

        logger.debug(f"Using model: {model}")

        if model is None:
            logger.error(
                f"Model provisioning failed: No model found. "
                f"Selection reason: {selection_reason}. "
                f"model_id={model_id}, default_type={default_type}. "
                f"Please check Settings → Models and ensure a default model is configured for '{default_type}'."
            )
            raise ConfigurationError(
                f"No model configured for {selection_reason}. "
                f"Please go to Settings → Models and configure a default model for '{default_type}'."
            )

        if not isinstance(model, LanguageModel):
            logger.error(
                f"Model type mismatch: Expected LanguageModel but got {type(model).__name__}. "
                f"Selection reason: {selection_reason}. "
                f"model_id={model_id}, default_type={default_type}."
            )
            raise ConfigurationError(
                f"Model is not a LanguageModel: {model}. "
                f"Please check that the model configured for '{default_type}' is a language model, not an embedding or speech model."
            )

        langchain_model = model.to_langchain()
        _model_cache[key] = (time.monotonic(), langchain_model)
        return langchain_model
