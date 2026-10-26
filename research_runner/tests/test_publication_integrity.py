"""CPU checks that cleanup attestations cannot bypass scientific integrity."""
from copy import deepcopy
import unittest

from research_runner.artifacts import IncompatibleResultsError, file_sha256, read_json, write_json
from research_runner.engine import ResearchRunner
from research_runner.tests import test_engine


class PublicationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_engine.EngineTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)
        self.fixture.prepare()
        self.runner = self.fixture.runner
        self.path = self.runner.store.root / 'dataset'
        self.manifest = read_json(self.path / 'manifest.json')
        self.config = deepcopy(self.fixture.config)
        # Simulate a different source inventory after removing legacy tests.
        self.config['version'] += 1
        self.upgraded = ResearchRunner(self.config, executor=self.fixture.executor)
        self.receipt = {
            'version': 1, 'name': 'dataset', 'parents': [],
            'from_signature': self.manifest['signature'],
            'to_signature': self.upgraded.signature('dataset'),
            'manifest_sha256': file_sha256(self.path / 'manifest.json'),
        }
        self.receipt_path = self.path.parent / '.compatibility/dataset.json'
        write_json(self.receipt_path, self.receipt)

    def test_exact_attestation_preserves_readability_and_original_manifest(self):
        self.assertEqual(self.upgraded.compatible('dataset'), self.path)
        self.assertEqual(read_json(self.path / 'manifest.json'), self.manifest)

    def test_attestation_rejects_scientific_config_or_receipt_drift(self):
        changed = deepcopy(self.config)
        changed['data']['reference_test_sha256'] = 'a' * 64
        with self.assertRaises(IncompatibleResultsError):
            ResearchRunner(changed).compatible('dataset')
        for key in ('to_signature', 'from_signature', 'manifest_sha256'):
            invalid = {**self.receipt, key: 'invalid'}
            write_json(self.receipt_path, invalid)
            with self.assertRaises(IncompatibleResultsError):
                self.upgraded.compatible('dataset')
        write_json(self.receipt_path, self.receipt)

    def test_attestation_does_not_allow_modified_result_bytes(self):
        indexed = next(iter(self.manifest['files']))
        path = self.path / indexed
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\ncorrupt')
            with self.assertRaises(ValueError):
                self.upgraded.compatible('dataset')
        finally:
            path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
