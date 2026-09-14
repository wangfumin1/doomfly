"""Summarize one local SNN baseline run without importing the neural runtime.

Expected layout under a run root::

    cue-screen/results.json
    conditioning-replication/results.json       # optional
    conditioning-strict/results.json            # optional when preflight blocks

The summary deliberately separates infrastructure/protocol failures from sensory
preflight and associative-learning failures. It makes no biological claim.
"""
import argparse
import json
from pathlib import Path


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _failed_checks(payload):
    if not payload:
        return []
    return sorted(name for name, passed in payload.get('checks', {}).items() if not passed)


def _strict_selectivities(payload):
    if not payload:
        return []
    rows = payload.get('rows', [])
    result = []
    for plus in (0, 1):
        subset = {row.get('condition'): row for row in rows if row.get('plus') == plus}
        result.append({
            'plus': plus,
            **{
                condition: subset.get(condition, {}).get('selectivity')
                for condition in ('paired', 'unpaired', 'frozen', 'no_imposed_US')
            },
        })
    return result


def summarize(root):
    root = Path(root)
    cue = _load(root / 'cue-screen/results.json')
    legacy = _load(root / 'conditioning-replication/results.json')
    strict = _load(root / 'conditioning-strict/results.json')

    cue_analysis = cue.get('analysis', {}) if cue else {}
    recommended = cue_analysis.get('recommended_pair') if cue else None
    preflight_passed = bool(cue and cue.get('conditioning_preflight_passed'))

    summary = {
        'run_root': str(root),
        'cue_screen': {
            'present': cue is not None,
            'passed': preflight_passed,
            'ready_pair_count': cue_analysis.get('ready_pair_count') if cue else None,
            'recommended_pair': recommended,
        },
        'legacy_replication': {
            'present': legacy is not None,
            'development_gate': legacy.get('development_gate') if legacy else None,
        },
        'strict_conditioning': {
            'present': strict is not None,
            'cue_names': strict.get('cue_names') if strict else None,
            'development_gate': strict.get('development_gate') if strict else None,
            'failed_checks': _failed_checks(strict),
            'selectivities': _strict_selectivities(strict),
        },
    }

    if not cue:
        status = 'incomplete'
        next_step = 'Run cue screening before any learning experiment.'
    elif not preflight_passed:
        status = 'blocked_no_balanced_cue'
        next_step = (
            'Do not tune plasticity. Revise sensory/cue encoding or candidate cues until '
            'at least one distinguishable, activity-balanced pair passes preflight.'
        )
    elif not strict:
        status = 'incomplete'
        next_step = 'Run strict conditioning using the preflight-selected cue pair.'
    elif strict.get('development_gate'):
        status = 'strict_gate_passed'
        next_step = (
            'Freeze parameters and repeat independent runs/cue pairs. Do not tune on Doom '
            'survival before replication establishes a robust associative effect.'
        )
    else:
        failed = _failed_checks(strict)
        integrity_markers = ('frozen_unchanged', 'memory_reset_restores', 'dose_matched', 'duration_matched')
        if any(any(marker in name for marker in integrity_markers) for name in failed):
            status = 'strict_protocol_or_state_failure'
            next_step = (
                'Fix protocol/state integrity before changing the learning rule; the current '
                'run cannot isolate associative plasticity.'
            )
        else:
            status = 'strict_learning_gate_failed'
            next_step = (
                'Keep the sensory pair and protocol fixed; next compare the current centered '
                'rule against an explicit three-factor/reward-modulated eligibility rule.'
            )

    summary['status'] = status
    summary['recommended_next_step'] = next_step
    return summary


def markdown(summary):
    cue = summary['cue_screen']
    strict = summary['strict_conditioning']
    lines = [
        '# SNN baseline run report',
        '',
        f"- Status: `{summary['status']}`",
        f"- Cue preflight passed: `{cue['passed']}`",
        f"- Conditioning-ready cue pairs: `{cue['ready_pair_count']}`",
    ]
    pair = cue.get('recommended_pair')
    if pair:
        lines.append(f"- Recommended cue pair: `{pair.get('cue_a')}` / `{pair.get('cue_b')}`")
    lines.extend([
        f"- Strict conditioning present: `{strict['present']}`",
        f"- Strict development gate: `{strict['development_gate']}`",
    ])
    if strict.get('failed_checks'):
        lines.append('- Failed strict checks:')
        lines.extend(f"  - `{name}`" for name in strict['failed_checks'])
    if strict.get('selectivities'):
        lines.extend(['', '## Selectivity'])
        for row in strict['selectivities']:
            lines.append(
                '- CS+ cue {plus}: paired={paired}, unpaired={unpaired}, '
                'frozen={frozen}, no-US={no_imposed_US}'.format(**row)
            )
    lines.extend([
        '',
        '## Next step',
        '',
        summary['recommended_next_step'],
        '',
        '> Development output only. A passed gate is not independent biological validation.',
        '',
    ])
    return '\n'.join(lines)


def write_report(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary = summarize(root)
    (root / 'research-report.json').write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'research-report.md').write_text(markdown(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    report = write_report(args.root)
    print(json.dumps(report, indent=2))
