import contextlib
import io
import unittest

from lminfer.cli import build_parser
from lminfer.config import EngineConfig
from lminfer.repair import repair_token_counts


class RepairWindowTest(unittest.TestCase):
    def test_token_counts(self):
        for length, begin, end, expected in [
            (100, 0.15, 0.2, (15, 20)),
            (100, 0.29, 0.57, (29, 57)),
            (13, 0.15, 0.2, (1, 2)),
            (3, 0.15, 0.15, (0, 0)),
            (100, 0, 0.2, (0, 20)),
            (100, 0.2, 0, (20, 0)),
            (100, 0.8, 0.7, (80, 20)),
            (100, 1, 1, (100, 0)),
            (0, 0.15, 0.15, (0, 0)),
        ]:
            with self.subTest(length=length, begin=begin, end=end):
                self.assertEqual(repair_token_counts(length, begin, end), expected)

    def test_cli_independent_ratios_and_defaults(self):
        parser = build_parser()
        args = parser.parse_args(['serve', 'model', '--repair-window-begin', '0.15',
                                  '--repair-window-end', '0.3'])
        self.assertEqual((args.repair_window_begin, args.repair_window_end), (0.15, 0.3))
        args = parser.parse_args(['serve', 'model'])
        self.assertEqual((args.repair_window_begin, args.repair_window_end), (0, 0))

    def test_invalid_ratios_rejected_by_cli_and_config(self):
        parser = build_parser()
        for edge in ('begin', 'end'):
            for value in ('-0.1', '1.1', 'nan', 'inf', '-inf'):
                with self.subTest(edge=edge, value=value):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parser.parse_args(['serve', 'model', f'--repair-window-{edge}={value}'])
                    with self.assertRaises(ValueError):
                        EngineConfig(model='model', **{f'repair_window_{edge}': float(value)})
