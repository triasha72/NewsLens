"""Request tracing and latency observability for NewsLens."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import (
    RequestResponseEndpoint,
)
from starlette.responses import Response

LOGGER = logging.getLogger("newslens.api")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
MAX_REQUEST_ID_LENGTH = 128


def _resolve_request_id(request: Request) -> str:
    """Use a supplied request ID or generate a new one."""

    supplied_request_id = request.headers.get(
        REQUEST_ID_HEADER,
        "",
    ).strip()

    if supplied_request_id:
        return supplied_request_id[:MAX_REQUEST_ID_LENGTH]

    return str(uuid4())


def get_request_id(request: Request) -> str:
    """Return the request ID assigned by middleware."""

    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    if request_id is None:
        return "unavailable"

    return str(request_id)


def install_request_observability(
    application: FastAPI,
) -> None:
    """Install request tracing and latency middleware."""

    @application.middleware("http")
    async def observe_request(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = _resolve_request_id(request)
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1_000

            LOGGER.exception(
                "request_failed request_id=%s method=%s path=%s duration_ms=%.3f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1_000

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.3f}"

        LOGGER.info(
            "request_completed request_id=%s method=%s path=%s status_code=%d duration_ms=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response
