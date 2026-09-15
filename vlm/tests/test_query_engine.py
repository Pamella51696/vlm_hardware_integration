import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import build_prompt
from query_engine import QueryEngine, _asked_side
from response_parser import extract_json_object, fallback, normalize


class ParserTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        text = 'Sure.\n```json\n{"answer":"yes","object":"curb","side":"left","confidence":0.91}\n```'
        parsed = extract_json_object(text)
        out = normalize(parsed, "Is there a curb on the left side?")
        self.assertEqual(out["answer"], "yes")
        self.assertEqual(out["object"], "curb")
        self.assertEqual(out["side"], "left")
        self.assertAlmostEqual(out["confidence"], 0.91)

    def test_fallback_has_schema(self):
        out = fallback("Is there a pedestrian?")
        self.assertEqual(out["answer"], "unknown")
        self.assertEqual(out["object"], "pedestrian")
        self.assertIn("confidence", out)

    def test_prompt_mentions_json(self):
        p = build_prompt("Is the road clear?", hint="Focus on the path", roi="center")
        self.assertIn("JSON", p)
        self.assertIn("Is the road clear?", p)


class EngineTests(unittest.TestCase):
    def test_asked_side(self):
        self.assertEqual(_asked_side("curb on the left", "full"), "left")
        self.assertEqual(_asked_side("anything", "right"), "right")

    def test_pedestrian_does_not_crash(self):
        import cv2
        import numpy as np

        img = np.full((400, 1920, 3), 30, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        engine = QueryEngine(backend=None)
        result = engine.answer(buf.tobytes(), "Is there a pedestrian?")
        self.assertEqual(result["object"], "pedestrian")
        self.assertIn(result["answer"], {"yes", "no", "unknown"})

    def test_synthetic_blank_frame(self):
        import cv2
        import numpy as np

        blank = np.zeros((240, 480, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank)
        self.assertTrue(ok)
        engine = QueryEngine(backend=None)
        result = engine.answer(buf.tobytes(), "Is there a pedestrian?", roi="center")
        self.assertIn(result["answer"], {"yes", "no", "unknown"})
        self.assertEqual(result["object"], "pedestrian")
        self.assertIn("frame_id", result)

    def test_curb_like_line(self):
        import cv2
        import numpy as np

        img = np.full((240, 480, 3), 40, dtype=np.uint8)
        cv2.line(img, (10, 180), (140, 200), (200, 200, 200), 4)
        ok, buf = cv2.imencode(".jpg", img)
        engine = QueryEngine(backend=None)
        result = engine.answer(buf.tobytes(), "Is there a curb on the left side?", roi="left")
        self.assertEqual(result["object"], "curb")
        self.assertIn(result["answer"], {"yes", "no", "unknown"})


if __name__ == "__main__":
    unittest.main()
