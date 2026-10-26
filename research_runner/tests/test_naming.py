"""Functional naming and frozen implementation contracts, without legacy files."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from research_runner.artifacts import read_json
from research_runner.config import ConfigError, DEFAULT_CONFIG, load_config
from research_runner.naming import paper_order
from research_runner.scoring import route

ROOT = Path(__file__).resolve().parents[2]


class NamingTests(unittest.TestCase):
    def test_mapping_and_rejection_of_old_runtime_orders(self):
        config = load_config(ROOT / 'config/experiments.yaml')
        self.assertEqual(config['naming'], DEFAULT_CONFIG['naming'])
        self.assertEqual(paper_order('Screen-Gate-Audit', config), 'L1-L2-L3')
        self.assertEqual(paper_order('Gate-Screen-Audit', config), 'L2-L1-L3')
        self.assertEqual(set(config['models']), {'encoder', 'autoencoder', 'screen', 'audit'})
        with self.assertRaises(ValueError):
            route([], {}, {}, ['L1'], .5, .5)
        for key in ('gate_recall_floor', 'screen_fpr_budget'):
            self.assertIn(key, config['training']['calibration'])

    def test_every_order_uses_functional_names(self):
        config = load_config(ROOT / 'config/experiments.yaml')
        row = {'id': 'x', 'original_trace_id': 'e', 'step_number': 1, 'label': 1}
        for order in config['experiments']['configuration']['orders']:
            output = route([row], {'x': {'score': 1}}, {'x': {'score': 0}},
                           order.split('-'), .5, .5, {'x': 1})[0]
            self.assertTrue({x['layer'] for x in output['route']} <= {'Screen', 'Gate', 'Audit'})
            self.assertIn('pending_audit', output)
            self.assertNotIn('pending_l3', output)

    def test_invalid_paper_mapping_is_rejected(self):
        config = yaml.safe_load((ROOT / 'config/experiments.yaml').read_text())
        config['naming']['paper']['Screen'] = 'L2'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ConfigError, 'naming must use version 2'):
                load_config(path)

    def test_production_imports_are_self_contained(self):
        forbidden = ('data', 'eval', 'training', 'models', 'experiments', 'utils', 'tools', 'config.config')
        for path in (ROOT / 'research_runner').rglob('*.py'):
            if 'tests' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or '']
                elif isinstance(node, ast.Import):
                    names = [name.name for name in node.names]
                for name in names:
                    self.assertFalse(any(name == p or name.startswith(p + '.') for p in forbidden),
                                     f'{path}:{node.lineno}: {name}')

    def test_payload_rule_semantics_match_frozen_origin(self):
        tree = ast.parse((ROOT / 'research_runner/implementation/data/data_preprocess.py').read_text())
        functions = {'normalize_retrieval', 'tool_name', 'normalize_trace',
                     'decoded_tool_content', 'audit_trace', 'channel_segment', 'expand_trace'}
        bodies = {node.name: ast.dump(node, include_attributes=False)
                  for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions}
        origins = read_json(ROOT / 'research_runner/implementation/origins.json')
        digest = hashlib.sha256(json.dumps(bodies, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, origins['payload_rules_semantic_sha256'])
        self.assertEqual(set(bodies), functions)


if __name__ == '__main__':
    unittest.main()
