"""Tests for Kaggle GPU lifecycle manager."""

import subprocess
from unittest.mock import MagicMock, patch, call
from contextlib import contextmanager

import pytest
import requests as _requests

from bud.lib.kaggle_gpu import (
    KaggleGPUConfig,
    KaggleGPUError,
    KaggleGPUManager,
    kaggle_gpu_session,
    _build_config,
)


def test_config_defaults():
    cfg = KaggleGPUConfig()
    assert cfg.kaggle_username == ""
    assert cfg.kernel_slug == "bud-gpu-server"
    assert cfg.ngrok_static_domain == ""
    assert cfg.llm_model == ""
    assert cfg.embed_model == ""
    assert cfg.poll_timeout_seconds == 600
    assert cfg.poll_interval_seconds == 15
    assert cfg.health_check_retries == 10
    assert cfg.model_cache_dataset == ""


def test_config_kernel_ref():
    cfg = KaggleGPUConfig(kaggle_username="alice", kernel_slug="my-kernel")
    assert cfg.kernel_ref == "alice/my-kernel"


def test_config_health_url():
    cfg = KaggleGPUConfig(ngrok_static_domain="example.ngrok-free.dev")
    assert cfg.health_url == "https://example.ngrok-free.dev"


def test_is_running_checks_health_at_static_domain():
    cfg = KaggleGPUConfig(ngrok_static_domain="example.ngrok-free.dev")
    manager = KaggleGPUManager(cfg)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("bud.lib.kaggle_gpu.requests.get", return_value=mock_resp) as mock_get:
        assert manager.is_running is True
    mock_get.assert_called_once_with("https://example.ngrok-free.dev", timeout=10)


def test_is_running_returns_false_on_failure():
    cfg = KaggleGPUConfig(ngrok_static_domain="example.ngrok-free.dev")
    manager = KaggleGPUManager(cfg)
    with patch("bud.lib.kaggle_gpu.requests.get",
               side_effect=_requests.exceptions.ConnectionError("timeout")):
        assert manager.is_running is False


def test_is_running_returns_false_when_no_domain():
    cfg = KaggleGPUConfig(ngrok_static_domain="")
    manager = KaggleGPUManager(cfg)
    assert manager.is_running is False


def test_stop_calls_kernels_cancel():
    cfg = KaggleGPUConfig(kaggle_username="alice", kernel_slug="my-kernel")
    manager = KaggleGPUManager(cfg)
    with patch("bud.lib.kaggle_gpu.subprocess.run") as mock_run:
        manager.stop()
    mock_run.assert_called_once_with(
        ["kaggle", "kernels", "cancel", "alice/my-kernel"],
        capture_output=True, text=True,
    )


def test_stop_ignores_errors():
    cfg = KaggleGPUConfig(kaggle_username="alice", kernel_slug="k")
    manager = KaggleGPUManager(cfg)
    with patch("bud.lib.kaggle_gpu.subprocess.run", side_effect=FileNotFoundError):
        manager.stop()  # Should not raise


def test_start_skips_push_when_already_running():
    cfg = KaggleGPUConfig(ngrok_static_domain="example.ngrok-free.dev")
    manager = KaggleGPUManager(cfg)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("bud.lib.kaggle_gpu.requests.get", return_value=mock_resp):
        with patch("bud.lib.kaggle_gpu.subprocess.run") as mock_run:
            url = manager.start()
    mock_run.assert_not_called()
    assert url == "https://example.ngrok-free.dev"


def test_start_pushes_kernel_and_waits():
    cfg = KaggleGPUConfig(
        kaggle_username="alice", kernel_slug="k",
        ngrok_static_domain="example.ngrok-free.dev",
        llm_model="test-llm", embed_model="test-embed",
        health_check_retries=2, poll_timeout_seconds=5, poll_interval_seconds=0,
    )
    manager = KaggleGPUManager(cfg)
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    health_call_count = 0
    def mock_get(url, timeout=10):
        nonlocal health_call_count
        health_call_count += 1
        if health_call_count <= 1:
            raise _requests.exceptions.ConnectionError("not up yet")
        return mock_resp_ok
    with patch("bud.lib.kaggle_gpu.requests.get", side_effect=mock_get):
        with patch("bud.lib.kaggle_gpu.subprocess.run") as mock_run:
            url = manager.start()
    push_calls = [c for c in mock_run.call_args_list if "push" in c.args[0]]
    assert len(push_calls) == 1
    assert url == "https://example.ngrok-free.dev"


def test_wait_for_health_timeout():
    cfg = KaggleGPUConfig(
        kaggle_username="alice", kernel_slug="k",
        ngrok_static_domain="example.ngrok-free.dev",
        poll_timeout_seconds=0, poll_interval_seconds=0,
    )
    manager = KaggleGPUManager(cfg)
    with patch("bud.lib.kaggle_gpu.requests.get",
               side_effect=_requests.exceptions.ConnectionError("down")):
        with patch("bud.lib.kaggle_gpu.subprocess.run"):
            with pytest.raises(KaggleGPUError, match="Timed out"):
                manager.start()


def test_push_kernel_kaggle_cli_not_found():
    cfg = KaggleGPUConfig(
        kaggle_username="alice", kernel_slug="k",
        ngrok_static_domain="example.ngrok-free.dev",
    )
    manager = KaggleGPUManager(cfg)
    with patch("bud.lib.kaggle_gpu.requests.get",
               side_effect=_requests.exceptions.ConnectionError("not up")):
        with patch("bud.lib.kaggle_gpu.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(KaggleGPUError, match="kaggle CLI not found"):
                manager.start()


def test_build_config_maps_keys():
    config = {
        "kaggle": {
            "username": "bob", "kernel_slug": "my-kernel",
            "ngrok_static_domain": "example.ngrok-free.dev",
            "poll_timeout_seconds": 120, "model_cache_dataset": "bob/cache",
        },
        "llm": {"model": "some-llm-model"},
        "embeddings": {"model": "some-embed-model"},
    }
    gpu_cfg = _build_config(config)
    assert gpu_cfg.kaggle_username == "bob"
    assert gpu_cfg.kernel_slug == "my-kernel"
    assert gpu_cfg.ngrok_static_domain == "example.ngrok-free.dev"
    assert gpu_cfg.poll_timeout_seconds == 120
    assert gpu_cfg.llm_model == "some-llm-model"
    assert gpu_cfg.embed_model == "some-embed-model"
    assert gpu_cfg.model_cache_dataset == "bob/cache"


def test_session_starts_and_stops():
    cfg = {"kaggle": {
        "username": "alice", "kernel_slug": "k",
        "ngrok_static_domain": "example.ngrok-free.dev",
    }}
    with patch("bud.lib.kaggle_gpu.KaggleGPUManager") as MockManager:
        mock_mgr = MockManager.return_value
        mock_mgr.start.return_value = "https://example.ngrok-free.dev"
        with kaggle_gpu_session(cfg):
            mock_mgr.start.assert_called_once()
        mock_mgr.stop.assert_called_once()


def test_session_stops_on_exception():
    cfg = {"kaggle": {
        "username": "alice", "kernel_slug": "k",
        "ngrok_static_domain": "example.ngrok-free.dev",
    }}
    with patch("bud.lib.kaggle_gpu.KaggleGPUManager") as MockManager:
        mock_mgr = MockManager.return_value
        mock_mgr.start.return_value = "https://example.ngrok-free.dev"
        with pytest.raises(ValueError):
            with kaggle_gpu_session(cfg):
                raise ValueError("boom")
        mock_mgr.stop.assert_called_once()


def test_session_noop_when_no_kaggle_config():
    cfg = {"llm": {"provider": "ollama"}}
    with kaggle_gpu_session(cfg):
        pass  # Should not raise
