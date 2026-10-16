"""Check public documentation, language, links and experiment artifacts."""
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def public_files():
    """Prune local environments/caches and historical acquisition snapshots before walking."""
    for directory, folders, names in os.walk(ROOT):
        folders[:] = [name for name in folders if not name.startswith('.')
                      and name not in {'__pycache__', 'source_snapshot'}]
        for name in names:
            if not name.startswith('.'):
                yield Path(directory) / name


def main():
    required = ['README.md', 'LICENSE', 'CITATION.cff', 'CHANGELOG.md',
                'docs/REPORT.md', 'docs/PROTOCOL.md', 'docs/REPRODUCTION.md',
                'docs/VALIDATION.md', 'evidence/release-validation.json',
                'notebooks/upgrade_demo.ipynb',
                'output/pdf/physbo_upgrade_report.pdf']
    errors = [f'missing {p}' for p in required if not (ROOT / p).is_file()]
    text_types = {'.md', '.py', '.json', '.ipynb', '.cff', '.toml', '.yml', '.yaml', '.txt'}
    files = list(public_files())
    for path in files:
        relative = path.relative_to(ROOT)
        if (not path.is_file() or path.suffix not in text_types
                or any(part.startswith('.') for part in relative.parts)
                or 'source_snapshot' in relative.parts):
            continue
        if re.search(r'[\u3400-\u9fff]', path.read_text(encoding='utf-8')):
            errors.append(f'{relative}: untranslated CJK text in the English project')
    for path in (p for p in files if p.suffix == '.md'):
        relative = path.relative_to(ROOT)
        if any(part.startswith('.') for part in relative.parts) or 'source_snapshot' in relative.parts:
            continue
        source = path.read_text(encoding='utf-8')
        for _, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', source):
            target = target.strip('<>')
            if '://' in target or target.startswith(('mailto:', '#')):
                continue
            target = unquote(target.split('#', 1)[0])
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f'{relative}: broken/outside link {target}')
        if re.search(r'[A-Za-z]:[/\\](?:Users|Research)[/\\]', source):
            errors.append(f'{relative}: machine-specific absolute path')
    out = ROOT / 'results/upgrade'
    jobs = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    if len(jobs) != 120:
        errors.append('formal manifest is not 120 jobs')
    for job in jobs:
        if not (out / 'runs' / (job['id'] + '.json')).is_file():
            errors.append(f'missing formal run {job["id"]}')
    for directory, expected in [('evidence/repeats', 6), ('evidence/smoke-working/runs', 6), ('evidence/smoke-isolated/runs', 6)]:
        if len(list((ROOT / directory).glob('*.json'))) != expected:
            errors.append(f'{directory}: expected {expected} records')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('PASS: English documentation, public links, 120 logs, repeat evidence and report assets')


if __name__ == '__main__':
    main()
