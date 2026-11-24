"""Reject secrets, private runtime files and oversized artifacts before committing."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'], cwd=root, text=True).splitlines()
secret = ''
env = root / '.env'
if env.exists():
    for line in env.read_text().splitlines():
        if line.startswith('GROQ_API_KEY='):
            secret = line.split('=', 1)[1].strip().strip('"').strip("'")
bad = []
for name in names:
    parts = Path(name).parts
    if any(p in ('data', '.venv', 'node_modules', '__pycache__', '.git') for p in parts) or (Path(name).name.startswith('.env') and Path(name).name != '.env.example'):
        bad.append(name)
        continue
    content = subprocess.check_output(['git', 'show', ':' + name], cwd=root)
    if len(content) > 5_000_000 or (secret and secret.encode() in content) or re.search(rb'(?:gsk_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})', content):
        bad.append(name)
if bad:
    raise SystemExit('Commit audit failed for: ' + ', '.join(bad))
print(f'PASS: {len(names)} staged files contain no configured key, private runtime folders, or oversized artifacts.')
