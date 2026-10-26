"""Small CPU-only publication checks; no real artifacts are changed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('prepare_zenodo', Path(__file__).with_name('prepare_zenodo.py'))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class PublicationTests(unittest.TestCase):
    def test_metadata_redaction_preserves_measurements_and_payloads(self):
        root = Path('/home/private/research')
        value = {'path': str(root / 'results/run'), 'f1': .9123,
                 'payload': '[PROMPT] Ordinary task', 'id': 'case_42',
                 'hostname': 'private-host', 'api_key': 'private-value'}
        output = export.clean(value, root)
        self.assertEqual(output['path'], '/workspace/ig-defense/results/run')
        self.assertEqual(output['f1'], value['f1'])
        self.assertEqual(output['payload'], value['payload'])
        self.assertEqual(output['id'], value['id'])
        self.assertEqual(output['hostname'], '[REDACTED]')
        self.assertEqual(output['api_key'], '[REDACTED]')
        self.assertEqual(value['hostname'], 'private-host')

    def test_private_path_in_evidence_and_credential_like_text_are_rejected(self):
        with self.assertRaises(ValueError):
            export.clean({'payload': '/home/private/research/secret'}, Path('/home/private/research'))
        with self.assertRaises(ValueError):
            export.clean({'text': 'sk-or-v1-' + 'x' * 40}, Path('/workspace/project'))

    def test_export_preserves_layout_and_sources_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / 'results/research_runner/dataset'
            group.mkdir(parents=True)
            (root / 'data/auditor_targeted').mkdir(parents=True)
            (root / 'publication').mkdir()
            (root / 'publication/README.md').write_text('Short description\n')
            original = json.dumps({'output': str(group), 'f1': .9})
            (group / 'metrics.json').write_text(original)
            (group / 'weights.safetensors').write_bytes(b'weights')
            (group / 'worker.log').write_text('private log')
            output = root / 'zenodo/ig-defense'
            with patch.object(export, 'GROUPS', ('dataset',)):
                report = export.prepare(root, output)
                with self.assertRaises(FileExistsError):
                    export.prepare(root, output)
            copied = output / 'results/research_runner/dataset/metrics.json'
            self.assertEqual(json.loads(copied.read_text())['f1'], .9)
            self.assertEqual((group / 'metrics.json').read_text(), original)
            self.assertFalse(copied.with_name('weights.safetensors').exists())
            self.assertFalse(copied.with_name('worker.log').exists())
            self.assertTrue((output / 'README.md').exists())
            self.assertTrue((output / 'EXPORT_MANIFEST.json').exists())
            for record in report['files']:
                self.assertEqual(export.sha(output / record['path']), record['export_sha256'])


if __name__ == '__main__':
    unittest.main()
