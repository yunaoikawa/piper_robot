"""Source-only paper analysis; no hardware or private event log required."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('door_complexity', ROOT / 'docs/evaluate_door_code_complexity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_counts_ignore_comments_docstrings_but_not_string_data():
    source = '''"""module documentation"""
# comment
import numpy as np
from numpy.linalg import norm as length

class A:
    """class documentation"""
    async def f(self):
        """method documentation"""
        def nested():
            return np.linalg.norm([1])
        value = "not a docstring"
        return length([1]) # same canonical API as above
'''
    result = module.count_source(source)
    assert result['code_lines_without_comments_or_docstrings'] == 8
    assert result['function_definitions'] == 2
    assert result['class_definitions'] == 1
    assert result['external_import_roots'] == ['numpy']
    assert result['external_api_names'] == ['numpy.linalg.norm']


def test_exact_patch_replay_and_fail_closed():
    path = module.SCOPE[0]
    files = {}
    module.apply_source_patch(files, f'*** Begin Patch\n*** Add File: {path}\n+def f():\n+    return 1\n*** End Patch')
    module.apply_source_patch(files, f'*** Begin Patch\n*** Update File: {path}\n@@ def f():\n-    return 1\n+    return 2\n*** End Patch')
    assert files[path] == 'def f():\n    return 2\n'
    with pytest.raises(ValueError, match='Missing hunk'):
        module.apply_source_patch(files, f'*** Begin Patch\n*** Update File: {path}\n@@\n-    return 99\n+    return 3\n*** End Patch')


def test_package_and_api_counts_are_unioned_across_files():
    result = module.summarize({'a.py': 'import numpy as np\nnp.zeros(1)\n',
                               'b.py': 'import numpy as np\nnp.zeros(2)\n'})
    assert result['external_package_count'] == result['external_api_name_count'] == 1
    assert result['python_files'] == 2


def test_tracked_snapshot_preserves_counts_and_git_checks():
    report = json.loads((ROOT / 'docs/assets/code_as_learning_machine/door_code_complexity_report.json').read_text())
    rows = report['configurations']
    assert [r['code_lines_without_comments_or_docstrings'] for r in rows] == [1238, 1312, 3094, 3108]
    assert [r['function_definitions'] for r in rows] == [36, 38, 85, 86]
    for row in rows:
        for key in ('function_definitions', 'code_lines_without_comments_or_docstrings'):
            assert row[key] == sum(m[key] for m in row['files'].values())
        assert set(row['files']) <= set(report['scope'])
    assert len(report['git_reconstruction_checks']) == 4
    assert all(c['all_scoped_sources_match'] for c in report['git_reconstruction_checks'])
