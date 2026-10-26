"""API integration tests with no GPU, model downloads, or tangram assets."""
import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from explain import cot
from translate import MockTranslator


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = tempfile.TemporaryDirectory()
        with patch.object(sys, "argv", ["server.py", "--model", "mock", "--via", "es,fr", "--logdir", cls.logs.name]), \
             patch("tangrams.load_set", return_value=["<svg/>"] * 12):
            cls.server = importlib.import_module("server")
        cls.server.app.testing = True

    @classmethod
    def tearDownClass(cls):
        cls.logs.cleanup()

    def setUp(self):
        self.s = self.server
        self.s.rooms.clear()
        self.s.modes = []
        self.s.mt = MockTranslator()
        self.client = self.s.app.test_client()
        self.room = self._testMethodName
        for role, lang in (("director", "en"), ("matcher", "zh")):
            self.assertEqual(self.post("/api/join", role=role, lang=lang).status_code, 200)

    def post(self, path, **data):
        return self.client.post(path, json=dict(room=self.room, **data))

    def debug(self):
        return self.client.get("/api/debug", query_string={"room": self.room}).get_json()

    def state(self, role):
        return self.client.get("/api/state", query_string={"room": self.room, "role": role}).get_json()

    def events(self):
        with open(os.path.join(self.logs.name, self.room + ".jsonl")) as f:
            return [json.loads(line) for line in f]

    def test_both_directions_and_player_visibility(self):
        self.assertEqual(self.post("/api/msg", role="director", text="a bird").status_code, 200)
        self.assertEqual(self.post("/api/msg", role="matcher", text="你好").status_code, 200)
        forward, reply = self.debug()["messages"]
        self.assertEqual(forward["route"], ["en", "es", "fr", "zh"])
        self.assertEqual(reply["route"], ["zh", "fr", "es", "en"])
        self.assertEqual([m["id"] for m in (forward, reply)], [0, 1])
        self.assertEqual(forward["hops"][1]["input"], forward["hops"][0]["text"])
        self.assertEqual(self.state("director")["messages"][0]["text"], "a bird")
        received = self.state("matcher")["messages"][0]
        self.assertEqual(received["text"], forward["hops"][-1]["text"])
        self.assertNotIn("hops", received)
        self.assertNotIn("orig", received)
        self.assertEqual(received["translation_steps"], 3)
        records = [e for e in self.events() if e["event"] == "message"]
        self.assertEqual(records[0]["hops"], forward["hops"])

    def test_configuration_changes_only_future_messages_and_survives_reset(self):
        self.post("/api/msg", role="director", text="first")
        self.assertEqual(self.post("/api/relay", via=["ja", "de"]).status_code, 200)
        self.post("/api/msg", role="director", text="second")
        before, after = self.debug()["messages"]
        self.assertEqual(before["via"], ["es", "fr"])
        self.assertEqual(after["via"], ["ja", "de"])
        self.assertEqual(self.post("/api/reset").status_code, 200)
        self.assertEqual(self.debug()["config"]["via"], ["ja", "de"])
        self.assertEqual(self.debug()["messages"], [])
        self.assertEqual([e for e in self.events() if e["event"] == "relay_changed"][0]["via"], ["ja", "de"])

    def test_direct_mode_and_same_language_relay(self):
        self.post("/api/relay", via=[])
        self.post("/api/msg", role="director", text="direct")
        self.assertEqual(self.debug()["messages"][-1]["shown"], "[en->zh] direct")
        self.post("/api/join", role="matcher", lang="en")
        self.post("/api/relay", via=["fr", "de"])
        self.post("/api/msg", role="director", text="round trip")
        self.assertEqual(self.debug()["messages"][-1]["route"], ["en", "fr", "de", "en"])

    def test_invalid_relay_does_not_change_configuration(self):
        for via in (None, "es,fr", ["xx"], [42], ["fr"] * 9):
            with self.subTest(via=via):
                self.assertEqual(self.post("/api/relay", via=via).status_code, 400)
                self.assertEqual(self.debug()["config"]["via"], ["es", "fr"])
        self.assertEqual(self.client.get("/api/meta").get_json()["default_via"], ["es", "fr"])

    def test_backtranslation_and_explanations_use_final_output(self):
        self.s.modes = ["backtranslation", "confidence", "alternatives", "history"]
        self.post("/api/msg", role="director", text="the bird")
        self.post("/api/msg", role="director", text="the bird again")
        m = self.s.rooms[self.room]["messages"][-1]
        self.assertEqual(m["back"], "[zh->en] " + m["shown"])
        self.assertEqual(m["explain"]["confidence"]["words"], m["hops"][-1]["words"])
        self.assertEqual(m["explain"]["alternatives"]["alts"], m["hops"][-1]["alts"])
        self.assertTrue(m["explain"]["history"]["links"])
        self.assertTrue(all(not h["alts"] for h in m["hops"][:-1]))
        with patch.object(self.s.mt, "explain", return_value="final-step rationale") as explain:
            result = cot(m, [], self.s.mt)
        self.assertEqual(explain.call_args.args, (m["hops"][-1]["input"], "fr", "zh", m["shown"]))
        self.assertEqual((result["step"], result["total"], result["done"]), (3, 3, True))
        self.assertNotIn("back", self.state("matcher")["messages"][-1])

    def test_injected_error_enters_only_the_first_step(self):
        with patch.object(self.s, "maybe_perturb", return_value=("right bird", ("left", "right"))) as perturb:
            self.post("/api/msg", role="director", text="left bird")
        perturb.assert_called_once()
        m = self.debug()["messages"][-1]
        self.assertEqual(m["orig"], "left bird")
        self.assertEqual(m["hops"][0]["input"], "right bird")
        self.assertEqual(m["hops"][1]["input"], m["hops"][0]["text"])

    def test_failed_translation_clears_pending_and_can_be_retried(self):
        original = self.s.mt.translate

        def fail(text, src, tgt, num_alts=0):
            if src == "es":
                raise RuntimeError("model failure")
            return original(text, src, tgt, num_alts)

        with patch.object(self.s.mt, "translate", side_effect=fail), self.assertLogs(self.s.app.logger, "ERROR"):
            response = self.post("/api/msg", role="director", text="bird")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.debug()["messages"], [])
        self.assertEqual(self.state("director")["pending"], [])
        failure = [e for e in self.events() if e["event"] == "translation_failed"][0]
        self.assertEqual(failure["step"], 2)
        self.assertEqual(self.post("/api/msg", role="director", text="bird").status_code, 200)

    def test_pending_translation_blocks_route_language_reset_and_round_changes(self):
        started, release = threading.Event(), threading.Event()
        original = self.s.mt.translate

        def blocking(text, src, tgt, num_alts=0):
            if src == "es":
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return original(text, src, tgt, num_alts)

        responses = []

        def send():
            with self.s.app.test_client() as client:
                responses.append(client.post("/api/msg", json=dict(room=self.room, role="director", text="bird")))

        with patch.object(self.s.mt, "translate", side_effect=blocking):
            thread = threading.Thread(target=send)
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                pending = self.state("matcher")["pending"][0]
                self.assertEqual((pending["step"], pending["total"]), (2, 3))
                self.assertIsNone(pending["text"])
                for path, data in (("/api/relay", {"via": []}), ("/api/reset", {}),
                                   ("/api/submit", {"role": "matcher"}),
                                   ("/api/join", {"role": "director", "lang": "fr"}),
                                   ("/api/msg", {"role": "director", "text": "duplicate"})):
                    self.assertEqual(self.post(path, **data).status_code, 409)
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(responses[0].status_code, 200)
        self.assertEqual(self.debug()["messages"][0]["round"], 1)
        self.assertEqual(self.debug()["messages"][0]["route"], ["en", "es", "fr", "zh"])

    def test_relay_persists_across_game_rounds(self):
        self.post("/api/msg", role="director", text="first round")
        self.post("/api/select", selection=self.debug()["target"])
        result = self.post("/api/submit", role="matcher").get_json()
        self.assertEqual(result["correct"], 12)
        self.post("/api/msg", role="director", text="second round")
        self.assertEqual(self.debug()["messages"][-1]["round"], 2)
        self.assertEqual(self.debug()["messages"][-1]["route"], ["en", "es", "fr", "zh"])

    def test_simultaneous_players_get_unique_ids_and_their_own_routes(self):
        ready = threading.Barrier(2)
        original = self.s.mt.translate

        def simultaneous(text, src, tgt, num_alts=0):
            if src in ("en", "zh"):
                ready.wait(timeout=5)
            return original(text, src, tgt, num_alts)

        responses = []

        def send(role):
            with self.s.app.test_client() as client:
                responses.append(client.post("/api/msg", json=dict(room=self.room, role=role, text="bird")))

        with patch.object(self.s.mt, "translate", side_effect=simultaneous):
            threads = [threading.Thread(target=send, args=(role,)) for role in ("director", "matcher")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(6)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual([r.status_code for r in responses], [200, 200])
        messages = self.debug()["messages"]
        self.assertEqual([m["id"] for m in messages], [0, 1])
        by_role = {m["from"]: m for m in messages}
        self.assertEqual(by_role["director"]["route"], ["en", "es", "fr", "zh"])
        self.assertEqual(by_role["matcher"]["route"], ["zh", "fr", "es", "en"])
        self.assertEqual(self.state("director")["pending"], [])

    def test_missing_partner_and_empty_message_are_rejected(self):
        self.assertEqual(self.post("/api/msg", role="director", text="  ").status_code, 400)
        self.assertEqual(self.post("/api/msg", role="other", text="hi").status_code, 400)
        self.s.rooms[self.room]["langs"].pop("matcher")
        self.assertEqual(self.post("/api/msg", role="director", text="hi").status_code, 409)


if __name__ == "__main__":
    unittest.main()
