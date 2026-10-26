"""Export a review copy without modifying source artifacts or creating a ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('dataset', 'training', 'configuration_stage1', 'configuration_stage2',
          'baselines', 'channel_ablation', 'backend', 'threshold', 'benign', 'auditor_targeted')
SKIP_PARTS = {'.staging', '.failed', '.progress', '.locks', '.runtime_smoke',
              '.compatibility', '__pycache__', 'progress'}
ALLOW = {'.json', '.jsonl', '.csv', '.md', '.txt', '.png', '.pdf', '.svg', '.tsv'}
PROTECTED = {'sequence', 'payload', 'prompt', 'content', 'reasoning_steps',
             'final_answer', 'system_prompt', 'evidence', 'evidence_span', 'text',
             'reason', 'user_intent'}
PRIVATE_KEYS = {'hostname', 'username', 'host_name', 'user_name', 'api_key',
                'access_token', 'auth_token', 'password', 'authorization'}
HOME_PATH = re.compile(r'/(?:home|Users)/[^/\s"\'\\]+')
SECRET = re.compile(r'(?:sk-or-v1-[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,})')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean(value, project, *, protected=False):
    """Change metadata paths only; never silently edit experimental text."""
    if isinstance(value, dict):
        return {clean(key, project, protected=protected): '[REDACTED]'
                if not protected and key.lower() in PRIVATE_KEYS and isinstance(item, str) and item
                else clean(item, project, protected=protected or key in PROTECTED)
                for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item, project, protected=protected) for item in value]
    if not isinstance(value, str):
        return value
    if SECRET.search(value):
        raise ValueError('Credential-like content found; manual review required (value not printed).')
    if protected:
        if str(project) in value or str(project.parent) + '/' in value:
            raise ValueError('Local account path in experimental text; refusing to alter evidence.')
        return value
    return HOME_PATH.sub('/home/USER', value.replace(str(project), '/workspace/ig-defense'))


def copy_public(source, target, project):
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    if source.suffix == '.json':
        original = source.read_text(encoding='utf-8')
        before = json.loads(original)
        after = clean(before, project)
        changed = after != before
        target.write_text(json.dumps(after, ensure_ascii=False, indent=2) + '\n'
                          if changed else original, encoding='utf-8')
    elif source.suffix == '.jsonl':
        with source.open(encoding='utf-8') as inp, target.open('w', encoding='utf-8') as out:
            for line in inp:
                if not line.strip():
                    out.write(line)
                    continue
                before = json.loads(line)
                after = clean(before, project)
                different = after != before
                changed |= different
                out.write(json.dumps(after, ensure_ascii=False, separators=(',', ':')) + '\n'
                          if different else line)
    elif source.suffix in {'.md', '.txt', '.csv', '.tsv', '.svg'}:
        before = source.read_text(encoding='utf-8')
        after = clean(before, project)
        changed = before != after
        target.write_text(after, encoding='utf-8')
    else:
        shutil.copyfile(source, target)
    return changed


def prepare(project=ROOT, output=None):
    project = Path(project).resolve()
    output = Path(output or project / 'zenodo/ig-defense').resolve()
    if output == project or project.is_relative_to(output):
        raise ValueError('Output must not contain the source project.')
    inputs = [project / 'results/research_runner' / group for group in GROUPS]
    inputs.append(project / 'data/auditor_targeted')
    for source in inputs:
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f'Missing or unsafe source directory: {source.relative_to(project)}')
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output overlaps a source directory.')
    if output.exists():
        raise FileExistsError('Export already exists; choose another --output. Nothing overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    records, omitted = [], []
    with tempfile.TemporaryDirectory(prefix='.zenodo-', dir=output.parent) as temp:
        stage = Path(temp) / 'ig-defense'
        stage.mkdir()
        for directory in inputs:
            for source in sorted(directory.rglob('*')):
                relative = source.relative_to(project)
                if source.is_symlink():
                    raise ValueError(f'Symlink requires manual review: {relative}')
                if not source.is_file():
                    continue
                if any(p in SKIP_PARTS for p in relative.parts) or source.suffix not in ALLOW:
                    omitted.append(relative.as_posix())
                    continue
                original_hash = sha(source)
                try:
                    redacted = copy_public(source, stage / relative, project)
                except ValueError as error:
                    raise ValueError(f'{relative}: {error}') from error
                if sha(source) != original_hash:
                    raise ValueError(f'Source changed during export: {relative}')
                records.append({'path': relative.as_posix(), 'source_sha256': original_hash,
                                'export_sha256': sha(stage / relative), 'metadata_redacted': redacted,
                                'bytes': (stage / relative).stat().st_size})
            print(f'Prepared {directory.relative_to(project)}', flush=True)
        shutil.copyfile(project / 'publication/README.md', stage / 'README.md')
        records.append({'path': 'README.md', 'export_sha256': sha(stage / 'README.md'),
                        'bytes': (stage / 'README.md').stat().st_size})
        manifest = {'purpose': 'public_review_snapshot_not_resumable', 'files': records,
                    'omitted': omitted, 'metadata_redacted_files': sum(r.get('metadata_redacted', False) for r in records)}
        (stage / 'EXPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        # Verify the prepared copy, not the immutable historical manifests.
        for record in records:
            if sha(stage / record['path']) != record['export_sha256']:
                raise ValueError('Export checksum validation failed.')
        stage.rename(output)
    print(f'Export ready: {output.name}; {len(records)} files; '
          f'{manifest["metadata_redacted_files"]} metadata files redacted. No ZIP created.', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    prepare(output=args.output)
