"""Unit tests for portal camera configuration loading; no live credentials or network."""
import io
import json
import logging
import os
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from main import PortalCameraLoadError, PortalCameraSupervisor, _request_door_access, _safe_event_api_error, load_portal_cameras
from camera_source import CameraConfig


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class PortalCameraLoadingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "PORTAL_API_URL": "http://localhost:3000",
            "RECOGNITION_WORKER_TOKEN": "  unit-test-token  ",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    @patch("urllib.request.urlopen")
    def test_loads_camera_and_sends_bearer_token(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps({"cameras": [{
            "id": "42", "owner_id": "7", "name": "Test Camera", "source": "rtsp://192.0.2.1/stream", "config_version": "2026-11-14T00:00:00.000Z",
        }]}).encode())

        cameras = load_portal_cameras()

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].id, "42")
        self.assertEqual(cameras[0].kind, "stream")
        self.assertEqual(cameras[0].owner_id, "7")
        self.assertEqual(urlopen.call_args.args[0].get_header("Authorization"), "Bearer unit-test-token")

    def test_worker_http_error_logs_safe_backend_diagnostic_without_secrets(self):
        digest = "a" * 64
        body = json.dumps({"code": "WORKER_SCOPE_CONFIG_INVALID", "error": f"Malformed Bearer dummy-token rtsp://user:pass@host/{digest}"}).encode()
        error = HTTPError("http://localhost:3000", 503, "Unavailable", {}, io.BytesIO(body))
        from main import _safe_worker_api_error
        safe = _safe_worker_api_error(error)
        self.assertIn("WORKER_SCOPE_CONFIG_INVALID", safe)
        self.assertNotIn("dummy-token", safe)
        self.assertNotIn("user:pass", safe)
        self.assertNotIn(digest, safe)

    @patch("urllib.request.urlopen", side_effect=lambda *_a, **_k: (_ for _ in ()).throw(HTTPError("http://localhost:3000", 401, "Unauthorized", {}, io.BytesIO())))
    def test_401_is_reported_as_authentication_failure(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "HTTP 401"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", side_effect=lambda *_a, **_k: (_ for _ in ()).throw(HTTPError("http://localhost:3000", 500, "Internal Server Error", {}, io.BytesIO())))
    def test_500_is_reported_as_internal_api_failure(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "HTTP 500"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", side_effect=TimeoutError())
    def test_timeout_is_reported(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "timed out"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", side_effect=URLError(ConnectionRefusedError()))
    def test_unreachable_portal_is_reported(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "unreachable"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", return_value=FakeResponse(b"not-json"))
    def test_invalid_json_is_reported(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "valid JSON"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", return_value=FakeResponse(b"\xff"))
    def test_non_utf8_json_is_reported(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "valid JSON"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", return_value=FakeResponse(b'{"cameras": {}}'))
    def test_invalid_response_shape_is_reported(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "invalid shape"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", return_value=FakeResponse(b'{"cameras": [{"id":"4","owner_id":"1","name":"Cam","source":"https://camera.test/","config_version":"v1"}]}'))
    def test_non_rtsp_camera_source_is_rejected(self, _urlopen):
        with self.assertRaisesRegex(PortalCameraLoadError, "unsupported camera source"):
            load_portal_cameras()

    @patch("urllib.request.urlopen", return_value=FakeResponse(b'{"cameras": []}'))
    def test_empty_camera_list_is_valid(self, _urlopen):
        self.assertEqual(load_portal_cameras(), [])

    def test_missing_token_is_reported_without_request(self):
        with patch.dict(os.environ, {"RECOGNITION_WORKER_TOKEN": ""}):
            with patch("urllib.request.urlopen") as urlopen:
                with self.assertRaisesRegex(PortalCameraLoadError, "not configured"):
                    load_portal_cameras()
                urlopen.assert_not_called()

    def test_event_api_error_body_is_sanitized_and_bounded(self):
        response_body = b'{"code":"VALIDATION","error":"Bearer dummy-secret rejected rtsp://user:pass@camera/stream"}'
        error = HTTPError("http://localhost:3000", 400, "Bad Request", {}, io.BytesIO(response_body))
        message = _safe_event_api_error(error)
        self.assertIn("VALIDATION", message)
        self.assertNotIn("dummy-secret", message)
        self.assertNotIn("user:pass", message)
        self.assertLessEqual(len(message), 300)

    @patch("urllib.request.urlopen")
    def test_recognized_event_requests_immediate_portal_door_authorization(self, urlopen):
        urlopen.return_value = FakeResponse(b'{"accepted":true,"granted":true,"reason":"dry_run"}')
        _request_door_access({"event_id": "evt-1", "camera_id": "42", "person_id": "9", "status": "recognized", "track_id": 2, "timestamp": "2026-11-15T00:00:00+00:00", "_tenant_id": "7"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:3000/api/door-access")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-token")
        payload = json.loads(request.data)
        self.assertNotIn("_tenant_id", payload)
        self.assertEqual(payload["person_id"], "9")

    @patch("urllib.request.urlopen")
    def test_unknown_recognition_never_requests_door_access(self, urlopen):
        _request_door_access({"event_id": "evt-2", "camera_id": "42", "person_id": None, "status": "unknown"})
        urlopen.assert_not_called()


class _FakeEvent:
    def __init__(self): self.set_called = False
    def set(self): self.set_called = True


class _FakeProcess:
    def __init__(self, *args, **kwargs): self.exitcode = None; self._alive = True
    def start(self): pass
    def is_alive(self): return self._alive
    def join(self, timeout=None): self._alive = False
    def terminate(self): self._alive = False


class _FakeContext:
    Event = _FakeEvent
    Process = _FakeProcess


class PortalSupervisorTests(unittest.TestCase):
    def camera(self, source="rtsp://192.0.2.1/stream", version="v1"):
        return CameraConfig(id="42", name="Cam", source=source, kind="stream", owner_id="7", config_version=version)

    def test_add_disable_and_change_reconcile_without_duplicates(self):
        supervisor = PortalCameraSupervisor(_FakeContext(), False, start_target=lambda *_args: None)
        supervisor.reconcile([self.camera()])
        self.assertEqual(list(supervisor.procs), ["42"])
        first = supervisor.procs["42"][1]
        supervisor.reconcile([self.camera()])
        self.assertIs(supervisor.procs["42"][1], first)
        supervisor.reconcile([self.camera(source="rtsp://192.0.2.1/new", version="v2")])
        self.assertIsNot(supervisor.procs["42"][1], first)
        supervisor.reconcile([])
        self.assertEqual(supervisor.procs, {})


if __name__ == "__main__":
    unittest.main()
