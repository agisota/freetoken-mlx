import pytest


pytest.importorskip("flashlib", reason="server defaults require the Linux/CUDA flashlib runtime")

from freetoken.server.args import ServerArgs


def test_default_server_boundary_is_loopback_with_local_cors_allowlist():
    assert ServerArgs.server_host == "127.0.0.1"
    assert "*" not in ServerArgs.cors_origins
    assert "tauri://localhost" in ServerArgs.cors_origins
