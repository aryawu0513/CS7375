import unittest

from relay import MAX_VIA, translate_chain, translation_route, validate_via
from translate import MockTranslator


class RecordingTranslator(MockTranslator):
    def __init__(self):
        self.calls = []

    def translate(self, text, src, tgt, num_alts=0):
        self.calls.append((text, src, tgt, num_alts))
        return super().translate(text, src, tgt, num_alts)


class RelayTests(unittest.TestCase):
    def test_forward_and_reverse_routes(self):
        self.assertEqual(translation_route("en", "zh", ["es", "fr"]), ["en", "es", "fr", "zh"])
        self.assertEqual(translation_route("zh", "en", ["es", "fr"], reverse=True), ["zh", "fr", "es", "en"])

    def test_direct_and_same_language_routes(self):
        self.assertEqual(translation_route("en", "zh", []), ["en", "zh"])
        self.assertEqual(translation_route("en", "en", []), ["en", "en"])
        self.assertEqual(translation_route("en", "en", ["fr", "de"]), ["en", "fr", "de", "en"])

    def test_only_adjacent_duplicate_languages_are_skipped(self):
        self.assertEqual(translation_route("en", "zh", ["en", "es", "es", "en", "zh"]),
                         ["en", "es", "en", "zh"])

    def test_invalid_configuration(self):
        for via in (None, "es,fr", ["xx"], [None], ["en", ["es"]], ["es"] * (MAX_VIA + 1)):
            with self.subTest(via=via), self.assertRaises(ValueError):
                validate_via(via)
        with self.assertRaises(ValueError):
            translation_route("invalid", "en", [])
        for route in ([], ["en"], ["en", "xx"]):
            with self.subTest(route=route), self.assertRaises(ValueError):
                translate_chain(MockTranslator(), "hello", route)

    def test_outputs_feed_next_step_and_only_final_step_requests_alternatives(self):
        mt, progress = RecordingTranslator(), []
        result, hops = translate_chain(mt, "bird", ["en", "es", "fr", "zh"], 2,
                                       on_step=lambda *args: progress.append(args))
        self.assertEqual(mt.calls, [("bird", "en", "es", 0),
                                    ("[en->es] bird", "es", "fr", 0),
                                    ("[es->fr] [en->es] bird", "fr", "zh", 2)])
        self.assertEqual(result["text"], "[fr->zh] [es->fr] [en->es] bird")
        self.assertEqual(result["words"], hops[-1]["words"])
        self.assertEqual(len(result["alts"]), 2)
        self.assertEqual(progress, [(1, 3, "en", "es"), (2, 3, "es", "fr"), (3, 3, "fr", "zh")])
        self.assertTrue(all(h["latency"] >= 0 for h in hops))

    def test_direct_translation_preserves_existing_result(self):
        mt = MockTranslator()
        for src, tgt in (("en", "zh"), ("en", "en")):
            result, hops = translate_chain(mt, "hello 世界", [src, tgt], 2)
            self.assertEqual(result, mt.translate("hello 世界", src, tgt, 2))
            self.assertEqual(len(hops), 1)

    def test_failure_stops_the_chain(self):
        class BrokenTranslator(RecordingTranslator):
            def translate(self, text, src, tgt, num_alts=0):
                if src == "es":
                    raise RuntimeError("failed second step")
                return super().translate(text, src, tgt, num_alts)

        mt = BrokenTranslator()
        with self.assertRaises(RuntimeError):
            translate_chain(mt, "bird", ["en", "es", "fr", "zh"])
        self.assertEqual(len(mt.calls), 1)


if __name__ == "__main__":
    unittest.main()
