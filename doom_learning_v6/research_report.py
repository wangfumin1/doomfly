"""Summarize one local SNN research run without importing the neural runtime.

Expected layout under a run root::

    cue-screen/results.json
    conditioning-replication/results.json              # optional
    conditioning-strict/results.json                   # centered-v6
    conditioning-strict-eligibility/results.json       # optional phase-1 comparison

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


def _strict_summary(payload):
    return {
        'present': payload is not None,
        'plasticity_model': payload.get('plasticity_model') if payload else None,
        'cue_names': payload.get('cue_names') if payload else None,
        'development_gate': payload.get('development_gate') if payload else None,
        'failed_checks': _failed_checks(payload),
        'selectivities': _strict_selectivities(payload),
    }


def _has_integrity_failure(payload):
    markers = ('frozen_unchanged', 'memory_reset_restores', 'dose_matched', 'duration_matched')
    return any(any(marker in name for marker in markers) for name in _failed_checks(payload))


def summarize(root):
    root = Path(root)
    cue = _load(root / 'cue-screen/results.json')
    legacy = _load(root / 'conditioning-replication/results.json')
    strict = _load(root / 'conditioning-strict/results.json')
    eligibility = _load(root / 'conditioning-strict-eligibility/results.json')

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
        'strict_conditioning': _strict_summary(strict),
        'eligibility_candidate': _strict_summary(eligibility),
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
        next_step = 'Run centered-v6 strict conditioning using the preflight-selected cue pair.'
    elif strict.get('development_gate'):
        status = 'strict_gate_passed'
        next_step = (
            'Freeze centered-v6 parameters and repeat independent cue sets/variants before '
            'moving to a closed-loop virtual-world task.'
        )
    elif _has_integrity_failure(strict):
        status = 'strict_protocol_or_state_failure'
        next_step = (
            'Fix protocol/state integrity before changing the learning rule; the current '
            'run cannot isolate associative plasticity.'
        )
    elif eligibility is None:
        status = 'strict_learning_gate_failed'
        next_step = (
            'Keep sensory encoding and protocol fixed. Run the preregistered '
            'eligibility-ltd-v6 comparison on the same preflight cue pair.'
        )
    elif _has_integrity_failure(eligibility):
        status = 'eligibility_protocol_or_state_failure'
        next_step = (
            'Fix the candidate implementation/state integrity before comparing learning '
            'mechanisms; do not interpret its conditioning result yet.'
        )
    elif eligibility.get('development_gate'):
        status = 'eligibility_candidate_gate_passed'
        next_step = (
            'Freeze eligibility-LTD parameters and replicate across other preflight-ready '
            'cue pairs/variants before selecting it as the Phase-1 learning mechanism.'
        )
    else:
        status = 'registered_rules_failed'
        next_step = (
            'Both preregistered mechanisms failed the same controlled assay. Preserve this '
            'negative result and implement the next explicit local three-factor candidate '
            'without changing the sensory pair or protocol.'
        )

    summary['status'] = status
    summary['recommended_next_step'] = next_step
    return summary


def _append_model(lines, title, payload):
    lines.extend([
        '',
        f'## {title}',
        '',
        f"- Present: `{payload['present']}`",
        f"- Model: `{payload['plasticity_model']}`",
        f"- Development gate: `{payload['development_gate']}`",
    ])
    if payload.get('failed_checks'):
        lines.append('- Failed checks:')
        lines.extend(f"  - `{name}`" for name in payload['failed_checks'])
    if payload.get('selectivities'):
        lines.append('- Selectivity:')
        for row in payload['selectivities']:
            lines.append(
                '  - CS+ cue {plus}: paired={paired}, unpaired={unpaired}, '
                'frozen={frozen}, no-US={no_imposed_US}'.format(**row)
            )


def markdown(summary):
    cue = summary['cue_screen']
    lines = [
        '# SNN research run report',
        '',
        f"- Status: `{summary['status']}`",
        f"- Cue preflight passed: `{cue['passed']}`",
        f"- Conditioning-ready cue pairs: `{cue['ready_pair_count']}`",
    ]
    pair = cue.get('recommended_pair')
    if pair:
        lines.append(f"- Recommended cue pair: `{pair.get('cue_a')}` / `{pair.get('cue_b')}`")
    _append_model(lines, 'Centered v6 strict conditioning', summary['strict_conditioning'])
    _append_model(lines, 'Eligibility-LTD v6 comparison', summary['eligibility_candidate'])
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
