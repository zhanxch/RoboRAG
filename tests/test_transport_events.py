"""Versioned startup notices must never admit tool activity or runtime errors."""

from pathlib import Path
import tempfile
import unittest

from roborag.transports.codex_cli import validate_cli_events


class TransportEventTests(unittest.TestCase):
    def validate(self, events):
        with tempfile.TemporaryDirectory() as folder:
            return validate_cli_events(events, Path(folder))

    def notice(self):
        return {"type": "item.completed", "item": {"type": "error", "message":
            "Code Mode is unavailable because code-mode host is disabled. Falling back to direct tools; enable `features.code_mode_host` and install `codex-code-mode-host`."}}

    def completion(self):
        return {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}

    def test_new_startup_notice_allowed_only_before_turn(self):
        self.assertEqual(self.validate([self.notice(), {"type": "turn.started"}, self.completion()]),
                         {"input_tokens": 5, "output_tokens": 2})
        with self.assertRaises(RuntimeError):
            self.validate([{"type": "turn.started"}, self.notice(), self.completion()])

    def test_fallback_notice_does_not_allow_actual_tool_calls(self):
        with self.assertRaises(RuntimeError):
            self.validate([self.notice(), {"type": "turn.started"},
                           {"type": "item.completed", "item": {"type": "command_execution"}},
                           self.completion()])

    def test_unknown_startup_errors_and_failed_turns_rejected(self):
        with self.assertRaises(RuntimeError):
            self.validate([{"type": "item.completed", "item": {"type": "error", "message": "bad"}},
                           self.completion()])
        with self.assertRaises(RuntimeError):
            self.validate([self.notice(), {"type": "turn.failed"}])

    def test_recovered_network_retry_requires_exactly_one_successful_turn(self):
        retry = {"type": "error", "message":
                 "Reconnecting... 2/5 (stream disconnected before completion: tls handshake eof)"}
        self.validate([{"type": "turn.started"}, retry, self.completion()])
        for events in ([retry], [retry, {"type": "turn.failed"}, self.completion()],
                       [retry, self.completion(), self.completion()],
                       [{"type": "error", "message": "Unrecognized transport failure"}, self.completion()]):
            with self.assertRaises(RuntimeError):
                self.validate(events)

    def test_only_the_explicit_proxy_feature_notice_is_allowed(self):
        message = (
            "Under-development features enabled: respect_system_proxy. Under-development "
            "features are incomplete and may behave unpredictably. To suppress this warning, "
            "set `suppress_unstable_features_warning = true` in /tmp/user/.codex/config.toml."
        )
        notice = {"type": "item.completed", "item": {"type": "error", "message": message}}
        self.validate([notice, {"type": "turn.started"}, self.completion()])
        with self.assertRaises(RuntimeError):
            self.validate([{"type": "turn.started"}, notice, self.completion()])
        notice["item"]["message"] = message.replace("respect_system_proxy", "multi_agent")
        with self.assertRaises(RuntimeError):
            self.validate([notice, {"type": "turn.started"}, self.completion()])


if __name__ == "__main__":
    unittest.main()
