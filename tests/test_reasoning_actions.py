"""Regression coverage for LFM3 reasoning suffixes and exact BrowserGym actions."""
import os
import re
import unittest

from bulfm.config import Config
from bulfm.environment import Browser, parse_action


class ReasoningActions(unittest.TestCase):
    def test_observed_lfm3_output(self):
        self.assertEqual(parse_action("[13]</think>click('13')"), ("click('13')", True))

    def test_complete_reasoning_block(self):
        self.assertEqual(parse_action("<think>[13] is a button</think>click('13')"),
                         ("click('13')", True))

    def test_reasoning_code_is_never_executed(self):
        self.assertEqual(parse_action("<think>click('13')</think>"), ("noop()", False))

    def test_unclosed_reasoning_is_invalid(self):
        self.assertEqual(parse_action("<think>click('13')"), ("noop()", False))

    def test_final_segment_still_requires_one_action(self):
        self.assertEqual(parse_action("[13]</think>click('13'); click('14')"), ("noop()", False))

    def test_final_segment_rejects_arbitrary_code(self):
        self.assertEqual(parse_action("[13]</think>__import__('os').system('id')"),
                         ("noop()", False))

    def test_delimiter_inside_valid_fill_text_is_preserved(self):
        text = "fill('2', 'literal </think> text')"
        self.assertEqual(parse_action(text), (text, True))

    @unittest.skipUnless(os.environ.get("BULFM_REAL_BROWSER") == "1", "Real browser opt-in")
    def test_reasoning_action_completes_real_click_test(self):
        env = Browser(Config())
        try:
            observation = env.reset(23)
            buttons = re.findall(r"\[(\d+)\] button", observation["tree"])
            self.assertIn("13", buttons)
            action, valid = parse_action("[13]</think>click('13')")
            self.assertTrue(valid)
            _, reward, done = env.step(action)
            self.assertGreater(reward, 0)
            self.assertTrue(done)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
