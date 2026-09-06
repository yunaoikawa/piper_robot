#!/usr/bin/env python3
"""Offline reconstruction/counting of Door task source from archived patches.

Never executes archived commands or writes reconstructed source into the repo.
The private event log is an input, not a redistributable paper artifact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tokenize
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path('/home/admin/.codex/sessions/2026/08/06/rollout-2026-08-06T09-12-22-019fd469-f996-7680-be0b-cf63eca2a112.jsonl')
SCOPE = (
    'rollout/incubator_door_demo.py',
    'rollout/incubator_door_visual.py',
    'rollout/incubator_door_plane.py',
    'src/compile_incubator_door_demos.py',
    'src/estimate_incubator_door_plane.py',
    'src/run_incubator_door_demo.py',
    'rollout/articulated_appliance.py',
    'src/run_incubator_door_autonomy.py',
)
CHECKPOINTS = ('2b6ccab', 'bfa9b7e', 'f2149bb', '283b913')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def patch_events(log):
    calls, outputs = {}, {}
    for line in log.open():
        # Limit parsing to the relevant date; never publish surrounding content.
        if '2026-08-08T' not in line[:100]:
            continue
        item = json.loads(line)
        stamp = item.get('timestamp', '')
        if not '2026-08-08T03:00' <= stamp <= '2026-08-08T10:10':
            continue
        payload = item.get('payload', {})
        kind = payload.get('type')
        if kind in ('custom_tool_call_output', 'function_call_output'):
            outputs[payload['call_id']] = (stamp, str(payload.get('output', '')))
        if kind not in ('custom_tool_call', 'function_call'):
            continue
        raw = payload.get('input', payload.get('arguments', ''))
        start = raw.find('"*** Begin Patch')
        if start < 0 or not any(Path(p).name in raw for p in SCOPE):
            continue
        # The recorded calls use a JSON-escaped string literal passed to apply_patch.
        patch, _ = json.JSONDecoder().raw_decode(raw[start:])
        calls[payload['call_id']] = (stamp, patch)
    events, rejected = [], []
    for call_id, (stamp, patch) in calls.items():
        if call_id not in outputs:
            raise ValueError(f'Missing patch acknowledgement: {call_id}')
        end, output = outputs[call_id]
        if 'Script failed' in output or 'verification failed' in output:
            rejected.append({'call_id': call_id, 'timestamp': stamp})
            continue
        if 'Script completed' not in output:
            raise ValueError(f'Unknown patch acknowledgement: {call_id}')
        events.append({'timestamp': end, 'call_id': call_id, 'patch': patch})
    return sorted(events, key=lambda e: e['timestamp']), rejected


def apply_source_patch(files, patch):
    """Apply supported exact-match Codex patch hunks in memory, failing closed."""
    lines = patch.splitlines()
    i = 1
    touched = []
    while i < len(lines) and lines[i] != '*** End Patch':
        header = lines[i]
        if not header.startswith(('*** Add File: ', '*** Update File: ')):
            raise ValueError(f'Unsupported patch header: {header}')
        mode, path = header[4:].split(': ', 1)
        path = path.removeprefix(str(ROOT) + '/')
        i += 1
        end = i
        while end < len(lines) and not lines[end].startswith('*** '):
            end += 1
        body = lines[i:end]
        i = end
        if path not in SCOPE:
            continue
        touched.append(path)
        if mode == 'Add File':
            if path in files or any(not line.startswith('+') for line in body):
                raise ValueError(f'Invalid add: {path}')
            files[path] = '\n'.join(line[1:] for line in body) + '\n'
            continue
        current = files[path].splitlines()
        cursor, j = 0, 0
        while j < len(body):
            anchor = body[j]
            if not anchor.startswith('@@'):
                raise ValueError(f'Expected hunk in {path}: {anchor}')
            if anchor.startswith('@@ '):
                seek = anchor[3:]
                positions = [k for k in range(cursor, len(current)) if current[k] == seek]
                if not positions:
                    raise ValueError(f'Missing anchor in {path}: {seek}')
                cursor = positions[0] + 1
            j += 1
            old, new = [], []
            while j < len(body) and not body[j].startswith('@@'):
                line = body[j]
                if line.startswith(' '):
                    old.append(line[1:]); new.append(line[1:])
                elif line.startswith('-'):
                    old.append(line[1:])
                elif line.startswith('+'):
                    new.append(line[1:])
                else:
                    raise ValueError(f'Unsupported hunk line in {path}: {line}')
                j += 1
            positions = [k for k in range(cursor, len(current) - len(old) + 1)
                         if current[k:k + len(old)] == old]
            if not positions:
                raise ValueError(f'Missing hunk in {path}: {old[:2]}')
            pos = positions[0]
            current[pos:pos + len(old)] = new
            cursor = pos + len(new)
        files[path] = '\n'.join(current) + '\n'
    return touched


def count_source(source):
    # The historical formatter removed redundant trailing blank lines.
    source = source.rstrip('\n') + '\n'
    tree = ast.parse(source)
    docstring_ranges = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstring_ranges.update(range(value.lineno, value.end_lineno + 1))
    code_lines = set()
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in ignored:
            code_lines.update(range(token.start[0], token.end[0] + 1))
    imports, aliases = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split('.')[0] for a in node.names)
            for alias in node.names:
                aliases[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split('.')[0])
            for alias in node.names:
                if alias.name != '*':
                    aliases[alias.asname or alias.name] = node.module + '.' + alias.name
    external = imports - sys.stdlib_module_names - {'__future__', 'rollout', 'robot', 'src'}
    api_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee, attributes = node.func, []
        while isinstance(callee, ast.Attribute):
            attributes.insert(0, callee.attr)
            callee = callee.value
        if isinstance(callee, ast.Name) and callee.id in aliases:
            canonical = aliases[callee.id]
            if canonical.split('.')[0] in external:
                api_names.add('.'.join([canonical] + attributes))
    return {
        'physical_lines': len(source.splitlines()),
        'code_lines_without_comments_or_docstrings': len(code_lines - docstring_ranges),
        'function_definitions': sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
        'class_definitions': sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        'external_import_roots': sorted(external),
        'external_api_names': sorted(api_names),
        'source_sha256': sha(source),
    }


def summarize(files):
    per_file = {name: count_source(text) for name, text in sorted(files.items())}
    external = sorted({name for metrics in per_file.values() for name in metrics['external_import_roots']})
    api_names = sorted({name for metrics in per_file.values() for name in metrics['external_api_names']})
    return {
        'python_files': len(per_file),
        **{key: sum(m[key] for m in per_file.values()) for key in (
            'physical_lines', 'code_lines_without_comments_or_docstrings',
            'function_definitions', 'class_definitions')},
        'external_import_roots': external, 'external_package_count': len(external),
        'external_api_names': api_names, 'external_api_name_count': len(api_names),
        'files': per_file,
    }


def evaluate(log):
    events, rejected = patch_events(log)
    report = json.loads((ROOT / 'docs/assets/code_as_learning_machine/door_first_approach_report.json').read_text())
    boundaries = []
    for row in report['configurations']:
        if row['status'] != 'measured':
            continue
        motion = json.loads((ROOT / row['motion']).read_text())
        stamp = datetime.fromtimestamp(motion['before']['timestamp_s'], timezone.utc).isoformat().replace('+00:00', 'Z')
        boundaries.append((stamp, 'run', row))
    for commit in CHECKPOINTS:
        stamp = subprocess.check_output(['git', 'show', '-s', '--format=%cI', commit], cwd=ROOT, text=True).strip()
        stamp = datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
        boundaries.append((stamp, 'commit', commit))
    files, applied, runs, checks = {}, [], [], []
    index = 0
    for stamp, kind, data in sorted(boundaries):
        while index < len(events) and events[index]['timestamp'] < stamp:
            event = events[index]
            touched = apply_source_patch(files, event['patch'])
            if touched:
                applied.append({'timestamp': event['timestamp'], 'call_id': event['call_id'],
                                'patch_sha256': sha(event['patch']), 'files': touched})
            index += 1
        if kind == 'commit':
            for path in SCOPE:
                result = subprocess.run(['git', 'show', f'{data}:{path}'], cwd=ROOT, text=True, capture_output=True)
                if result.returncode:
                    if path in files:
                        raise ValueError(f'Unexpected reconstructed file at {data}: {path}')
                elif files.get(path, '').rstrip('\n') != result.stdout.rstrip('\n'):
                    raise ValueError(f'Reconstruction differs from Git {data}:{path}')
            checks.append({'commit': data, 'timestamp': stamp, 'all_scoped_sources_match': True,
                           'normalization': 'Ignore redundant trailing newlines only; all other source bytes must match.'})
        else:
            runs.append({'display_stage': f'D{len(runs)+1}', 'historical_stage': data['stage'],
                         'snapshot_before_approach_utc': stamp, 'motion_path': data['motion'],
                         'applied_patch_count': len(applied), **summarize(files)})
    return {
        'schema': 'door_source_complexity/v1',
        'scope': list(SCOPE),
        'definitions': {
            'scope': 'All existing Door-specific Python source files in the fixed allowlist, including diagnostic branches and compilation; excludes tests, configs, shared robot infrastructure, dependencies, data and archived one-off commands.',
            'code_lines': 'Physical lines containing Python tokens, excluding comments, blank lines and module/class/function docstring lines.',
            'physical_lines': 'Physical lines after normalizing redundant trailing newlines; comments and docstrings included.',
            'functions': 'AST FunctionDef plus AsyncFunctionDef, including methods/nested functions; excludes lambdas.',
            'external_packages': 'Distinct directly imported non-standard-library root names in scoped source; static dependencies, NOT executed tools or transitive packages.',
            'external_api_names': 'Distinct syntactic call targets rooted in imported external aliases, canonicalized (e.g. np.linalg.norm -> numpy.linalg.norm). Not runtime invocations or dynamically resolved instance methods; includes diagnostic code.',
            'external_classification': 'Uses Python sys.stdlib_module_names and excludes repository roots rollout, robot and src.',
            'interpretation': 'Source size, not semantic or algorithmic complexity. Snapshot is before each selected approach; later trial patches are not backdated.',
        },
        'configurations': runs, 'git_reconstruction_checks': checks,
        'applied_patches': applied, 'rejected_patch_calls': rejected,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-log', type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.event_log), indent=2, allow_nan=False))
