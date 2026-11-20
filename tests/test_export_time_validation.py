"""from_time and to_time are validated before any HTTP request is sent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import SecretStr

from proofline import AsyncProoflineClient, ProoflineClient
from proofline import client as client_module

from .fixtures import ORG, TOKEN
from .http_server import Reply, Request, Server

MODES = pytest.mark.parametrize("mode", ["sync", "async"])

Client = ProoflineClient | AsyncProoflineClient


@asynccontextmanager
async def session(mode: str, url: str) -> AsyncIterator[Client]:
    if mode == "sync":
        with ProoflineClient(SecretStr(TOKEN), base_url=url) as client:
            yield client
    else:
        async with AsyncProoflineClient(SecretStr(TOKEN), base_url=url) as client:
            yield client


async def _call_export(client: Client, **kwargs: Any) -> Any:
    result = client.export_reviews(ORG, "reviews", **kwargs)
    if hasattr(result, "__await__"):
        return await result
    return result


async def _first_iter(client: Client, **kwargs: Any) -> Any:
    it = client.iter_export_pages(ORG, "reviews", **kwargs)
    if hasattr(it, "__aiter__"):
        return await it.__anext__()
    return next(iter(it))  # type: ignore[arg-type]


def _valid_export_reply(request: Request, index: int) -> Reply:
    return Reply(
        body=b'{"dataset_version":"1"}\n',
        headers={
            "Content-Type": "application/x-ndjson",
            "X-Proofline-Export-Version": "1",
            "X-Proofline-Export-Settled-Until": "2026-10-25T00:00:00Z",
        },
    )


# ---------------------------------------------------------------------------
# parametrised invalid inputs
# ---------------------------------------------------------------------------

_INVALID_FROM: list[tuple[str, dict[str, Any]]] = [
    ("from_int", {"from_time": 20260101, "to_time": "2026-10-25T00:00:00Z"}),
    ("from_bad_str", {"from_time": "not-a-date", "to_time": "2026-10-25T00:00:00Z"}),
    ("from_naive", {"from_time": "2026-10-01 00:00:00", "to_time": "2026-10-25T00:00:00Z"}),
]

_INVALID_TO: list[tuple[str, dict[str, Any]]] = [
    ("to_none", {"from_time": "2026-10-01T00:00:00Z", "to_time": None}),
    ("to_bad_str", {"from_time": "2026-10-01T00:00:00Z", "to_time": "bad"}),
    ("to_naive", {"from_time": "2026-10-01T00:00:00Z", "to_time": "2026-10-25 00:00:00"}),
]

_REVERSED: list[tuple[str, dict[str, Any]]] = [
    (
        "from_after_to",
        {"from_time": "2026-10-25T00:00:00Z", "to_time": "2026-10-01T00:00:00Z"},
    ),
]

_ALL_INVALID = _INVALID_FROM + _INVALID_TO + _REVERSED


# ---------------------------------------------------------------------------
# export_reviews – raises ValueError before any HTTP request
# ---------------------------------------------------------------------------


@MODES
@pytest.mark.parametrize(
    ("label", "kwargs"), _ALL_INVALID, ids=[label for label, _ in _ALL_INVALID]
)
async def test_export_rejects_malformed_time_window(
    mode: str, label: str, kwargs: dict[str, Any]
) -> None:
    with Server(lambda req, idx: Reply(body={})) as server:
        async with session(mode, server.url) as client:
            with pytest.raises(ValueError):
                await _call_export(client, **kwargs)
        assert not server.requests, "no HTTP request should be sent for an invalid time window"


@MODES
async def test_export_accepts_valid_time_window(mode: str) -> None:
    with Server(_valid_export_reply) as server:
        async with session(mode, server.url) as client:
            page = await _call_export(
                client,
                from_time="2026-10-01T00:00:00Z",
                to_time="2026-10-25T00:00:00Z",
            )
        assert page.version == 1
        assert len(server.requests) == 1


# ---------------------------------------------------------------------------
# iter_export_pages – raises ValueError before any HTTP request
# ---------------------------------------------------------------------------


@MODES
@pytest.mark.parametrize(
    ("label", "kwargs"), _ALL_INVALID, ids=[label for label, _ in _ALL_INVALID]
)
async def test_export_iterator_rejects_malformed_time_window(
    mode: str, label: str, kwargs: dict[str, Any]
) -> None:
    with Server(lambda req, idx: Reply(body={})) as server:
        async with session(mode, server.url) as client:
            with pytest.raises(ValueError):
                await _first_iter(client, **kwargs)
        assert not server.requests, "no HTTP request should be sent for an invalid time window"


@MODES
async def test_export_iterator_accepts_valid_time_window(mode: str) -> None:
    with Server(_valid_export_reply) as server:
        async with session(mode, server.url) as client:
            page = await _first_iter(
                client,
                from_time="2026-10-01T00:00:00Z",
                to_time="2026-10-25T00:00:00Z",
            )
        assert page.version == 1
        assert len(server.requests) == 1


# ---------------------------------------------------------------------------
# _export_time_window unit tests (no network)
# ---------------------------------------------------------------------------


def test_export_time_window_raises_on_non_string_from() -> None:
    with pytest.raises(ValueError, match="from_time"):
        client_module._export_time_window(123, "2026-10-25T00:00:00Z")


def test_export_time_window_raises_on_non_string_to() -> None:
    with pytest.raises(ValueError, match="to_time"):
        client_module._export_time_window("2026-10-01T00:00:00Z", None)


def test_export_time_window_raises_on_unparseable_from() -> None:
    with pytest.raises(ValueError, match="from_time"):
        client_module._export_time_window("oops", "2026-10-25T00:00:00Z")


def test_export_time_window_raises_on_unparseable_to() -> None:
    with pytest.raises(ValueError, match="to_time"):
        client_module._export_time_window("2026-10-01T00:00:00Z", "oops")


def test_export_time_window_raises_on_naive_from() -> None:
    with pytest.raises(ValueError, match="from_time"):
        client_module._export_time_window("2026-10-01 00:00:00", "2026-10-25T00:00:00Z")


def test_export_time_window_raises_on_naive_to() -> None:
    with pytest.raises(ValueError, match="to_time"):
        client_module._export_time_window("2026-10-01T00:00:00Z", "2026-10-25 00:00:00")


def test_export_time_window_raises_when_from_after_to() -> None:
    with pytest.raises(ValueError, match="from_time must not be after to_time"):
        client_module._export_time_window("2026-10-25T00:00:00Z", "2026-10-01T00:00:00Z")


def test_export_time_window_accepts_equal_boundaries() -> None:
    result = client_module._export_time_window("2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z")
    assert result == ("2026-10-01T00:00:00Z", "2026-10-01T00:00:00Z")


def test_export_time_window_accepts_valid_window() -> None:
    result = client_module._export_time_window("2026-10-01T00:00:00Z", "2026-10-25T00:00:00Z")
    assert result == ("2026-10-01T00:00:00Z", "2026-10-25T00:00:00Z")


def test_export_time_window_accepts_plus_offset() -> None:
    result = client_module._export_time_window(
        "2026-10-01T05:30:00+05:30", "2026-10-25T05:30:00+05:30"
    )
    assert result == ("2026-10-01T05:30:00+05:30", "2026-10-25T05:30:00+05:30")


# ---------------------------------------------------------------------------
# Mixed-offset and same-instant ordering (the cases the lexicographic
# comparison got wrong — fixed by comparing parsed datetimes)
# ---------------------------------------------------------------------------


def test_export_time_window_accepts_mixed_offset_valid_window() -> None:
    # 10:00+05:30 == 04:30Z  <  05:00Z — genuinely earlier; must be accepted.
    result = client_module._export_time_window(
        "2026-10-01T10:00:00+05:30",
        "2026-10-01T05:00:00+00:00",
    )
    # Original strings returned unchanged for use as query parameters.
    assert result == ("2026-10-01T10:00:00+05:30", "2026-10-01T05:00:00+00:00")


def test_export_time_window_rejects_mixed_offset_reversed_window() -> None:
    # 10:00+00:00 > 05:00+05:30 == 23:30Z previous day — from is after to.
    with pytest.raises(ValueError, match="from_time must not be after to_time"):
        client_module._export_time_window(
            "2026-10-01T10:00:00+00:00",
            "2026-10-01T05:00:00+05:30",
        )


def test_export_time_window_accepts_same_instant_different_spelling() -> None:
    # Both represent the same UTC instant; from == to is allowed.
    # Lexicographically "+05:30" > "+00:00" so the old string comparison
    # would have raised ValueError — datetime comparison handles this correctly.
    result = client_module._export_time_window(
        "2026-10-01T05:30:00+05:30",  # == 00:00Z
        "2026-10-01T00:00:00+00:00",  # == 00:00Z
    )
    assert result == ("2026-10-01T05:30:00+05:30", "2026-10-01T00:00:00+00:00")


def test_export_time_window_accepts_same_instant_with_fractional_seconds() -> None:
    # Fractional seconds that land on the same instant as a whole-second form.
    result = client_module._export_time_window(
        "2026-10-01T00:00:00.000000+00:00",
        "2026-10-01T00:00:00+00:00",
    )
    assert result == ("2026-10-01T00:00:00.000000+00:00", "2026-10-01T00:00:00+00:00")


def test_export_time_window_rejects_same_instant_mixed_offset_reversed() -> None:
    # from is lexicographically *before* to (Z < +05:30 in ASCII), but
    # semantically from == to, so it must still be accepted (not rejected).
    # This is the edge case where lex order and time order agree on equality;
    # confirm we don't accidentally raise.
    result = client_module._export_time_window(
        "2026-10-01T00:00:00Z",  # 00:00 UTC
        "2026-10-01T05:30:00+05:30",  # also 00:00 UTC
    )
    assert result == ("2026-10-01T00:00:00Z", "2026-10-01T05:30:00+05:30")


def test_export_time_window_original_strings_preserved_mixed_offset() -> None:
    # Confirm the helper never normalises the strings it returns — both are
    # forwarded to the server exactly as the caller wrote them.
    from_raw = "2026-10-01T10:00:00+05:30"
    to_raw = "2026-10-25T12:00:00-04:00"
    from_ret, to_ret = client_module._export_time_window(from_raw, to_raw)
    assert from_ret is from_raw
    assert to_ret is to_raw
