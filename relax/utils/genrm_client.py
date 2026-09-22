# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GenRM Client for accessing GenRM service.

This module provides an async client to communicate with the GenRM service for
generative reward model evaluations.
"""

import asyncio
from typing import Dict, List, Optional

import httpx

from relax.utils.logging_utils import get_logger
from relax.utils.utils import get_serve_url


# Retry policy for transient network failures against the GenRM service.
# Judge calls are cheap (short generations) so a bounded retry with modest
# backoff is safe. 5xx and transport-level errors are retried; 4xx is a
# client bug and re-raised immediately.
_GENRM_MAX_ATTEMPTS = 3
_GENRM_INITIAL_BACKOFF_SEC = 0.5


logger = get_logger(__name__)

# Module-level singleton to avoid creating a new client per request
_genrm_client: Optional["GenRMClient"] = None


class GenRMClient:
    """Async client for GenRM service.

    Provides async methods to interact with the GenRM service for
    reward/preference evaluations. Uses httpx.AsyncClient to avoid blocking the
    event loop in async rollout contexts.
    """

    def __init__(self, service_url: Optional[str] = None, timeout: float = 1800.0):
        """Initialize GenRM client.

        Args:
            service_url: URL of the GenRM service. If None, will use get_serve_url("genrm")
            timeout: Request timeout in seconds
        """
        if service_url is None:
            service_url = get_serve_url("genrm")

        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        # Raise pool limits above httpx's default 100 — a reward step can fire
        # thousands of concurrent judge calls and a small pool serializes them;
        # keepalive_expiry >> the default 5s so reused connections aren't reaped
        # mid-burst.
        self._async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=2048, keepalive_expiry=600),
        )
        # Keep a sync client for health checks during init and non-async contexts
        self._sync_client = httpx.Client(timeout=timeout)

        logger.info(f"GenRMClient initialized with service URL: {self.service_url}")

    async def generate(
        self,
        messages: List[dict],
        sampling_params: Optional[Dict] = None,
        route_key: Optional[str] = None,
    ) -> str:
        """Async generate response for given chat messages.

        Takes OpenAI-style messages as input, sends to GenRM service,
        and returns the raw model response. The caller is responsible
        for formatting the messages and parsing the response.

        Args:
            messages: List of messages with role and content
                (e.g., [{"role": "user", "content": "..."}])
            sampling_params: Optional sampling parameters to override defaults.
                Supported keys: temperature, top_p, top_k, max_new_tokens.
                Example: {"temperature": 0.3, "top_p": 0.9}
            route_key: Selects which genRM instance to use, when the service
                hosts more than one (see --genrm-instances). Typically the
                name of the reward/scoring task making the call. Omit to use
                the service's sole instance.

        Returns:
            Raw response string from the GenRM model
        """
        url = f"{self.service_url}/generate"
        payload: Dict = {
            "messages": messages,
        }
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if route_key is not None:
            payload["route_key"] = route_key

        backoff = _GENRM_INITIAL_BACKOFF_SEC
        for attempt in range(1, _GENRM_MAX_ATTEMPTS + 1):
            try:
                resp = await self._async_client.post(url, json=payload)
                resp.raise_for_status()
                result = resp.json()
                return result.get("response", "")
            except httpx.HTTPStatusError as e:
                # 4xx is a client bug — don't retry.
                if e.response.status_code < 500:
                    logger.error(f"GenRM generate 4xx (not retrying): {e}")
                    raise
                if attempt >= _GENRM_MAX_ATTEMPTS:
                    logger.error(f"GenRM generate 5xx after {attempt} attempts: {e}")
                    raise
                logger.warning(
                    f"GenRM generate 5xx attempt {attempt}/{_GENRM_MAX_ATTEMPTS}, retrying in {backoff:.2f}s: {e}"
                )
            except httpx.HTTPError as e:
                # Transport-level errors (ReadError, ConnectError, timeouts, ...).
                if attempt >= _GENRM_MAX_ATTEMPTS:
                    logger.error(f"GenRM generate transport error after {attempt} attempts ({type(e).__name__}): {e}")
                    raise
                logger.warning(
                    f"GenRM generate transport error attempt {attempt}/{_GENRM_MAX_ATTEMPTS} "
                    f"({type(e).__name__}), retrying in {backoff:.2f}s: {e}"
                )
            except Exception as e:
                logger.error(f"Unexpected error in GenRM generate: {e}")
                raise

            await asyncio.sleep(backoff)
            backoff *= 2

    def health_check(self) -> Dict:
        """Check health status of GenRM service (sync, safe to call at init).

        Returns:
            Dictionary with status information
        """
        url = f"{self.service_url}/health"
        try:
            resp = self._sync_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_metrics(self) -> Dict:
        """Get metrics from GenRM service.

        Returns:
            Dictionary with service metrics
        """
        url = f"{self.service_url}/metrics"
        try:
            resp = self._sync_client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"GenRM metrics request failed: {e}")
            return {}

    def close(self):
        """Close both HTTP clients."""
        self._sync_client.close()
        try:
            self._async_client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_genrm_client(service_url: Optional[str] = None, timeout: float = 1800.0) -> GenRMClient:
    """Get or create a singleton GenRM client.

    Uses module-level caching to avoid creating a new HTTP client and
    health-check round trip on every call.

    Args:
        service_url: URL of the GenRM service
        timeout: Request timeout in seconds

    Returns:
        GenRMClient instance (cached singleton)
    """
    global _genrm_client
    if _genrm_client is None:
        _genrm_client = GenRMClient(service_url=service_url, timeout=timeout)
    return _genrm_client
