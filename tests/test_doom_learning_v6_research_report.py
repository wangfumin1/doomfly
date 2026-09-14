import json
from pathlib import Path

from doom_learning_v6.research_report import summarize


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def cue_payload(passed=True):
    pair = {
        'cue_a': 'vertical',
        'cue_b': 'horizontal',
        'conditioning_ready': passed,
    }
    return {
        'conditioning_preflight_passed': passed,
        'analysis': {
            'ready_pair_count': 1 if passed else 0,
            'recommended_pair': pair if passed else None,
        },
    }


def strict_payload(gate=False, failed=None):
    checks = {'cue_0_paired_positive': True, 'cue_1_paired_positive': True}
    for name in failed or []:
        checks[name] = False
    return {
        'cue_names': ['vertical', 'horizontal'],
        'development_gate': gate,
        'checks': checks,
        'rows': [
            {'plus': plus, 'condition': condition, 'selectivity': value}
            for plus in (0, 1)
            for condition, value in (
                ('paired', .2), ('unpaired', .05), ('frozen', 0.), ('no_imposed_US', .01)
            )
        ],
    }


def test_report_blocks_learning_when_visual_preflight_fails(tmp_path):
    write(tmp_path / 'cue-screen/results.json', cue_payload(False))
    result = summarize(tmp_path)
    assert result['status'] == 'blocked_no_balanced_cue'
    assert 'Do not tune plasticity' in result['recommended_next_step']


def test_report_routes_associative_failure_to_learning_rule_comparison(tmp_path):
    write(tmp_path / 'cue-screen/results.json', cue_payload(True))
    write(
        tmp_path / 'conditioning-strict/results.json',
        strict_payload(False, ['cue_0_paired_beats_unpaired']),
    )
    result = summarize(tmp_path)
    assert result['status'] == 'strict_learning_gate_failed'
    assert 'three-factor' in result['recommended_next_step']


def test_report_does_not_hide_integrity_failure(tmp_path):
    write(tmp_path / 'cue-screen/results.json', cue_payload(True))
    write(
        tmp_path / 'conditioning-strict/results.json',
        strict_payload(False, ['cue_0_memory_reset_restores']),
    )
    result = summarize(tmp_path)
    assert result['status'] == 'strict_protocol_or_state_failure'


def test_report_requires_replication_after_pass(tmp_path):
    write(tmp_path / 'cue-screen/results.json', cue_payload(True))
    write(tmp_path / 'conditioning-strict/results.json', strict_payload(True))
    result = summarize(tmp_path)
    assert result['status'] == 'strict_gate_passed'
    assert 'independent' in result['recommended_next_step']
