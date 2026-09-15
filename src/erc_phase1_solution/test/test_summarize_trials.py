"""Tests for the dependency-light trial results summarizer."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import summarize_trials as summary  # noqa: E402


def _complete_summary(
    trial_id: str,
    *,
    success: bool = True,
    detected_row=2,
    **overrides,
):
    payload = {
        'trial_id': trial_id,
        'team_name': 'Test Team',
        'shelf_column_number': 3,
        'book_colour': 'blue',
        'dry_run': False,
        'success': success,
        'failure_reason': '' if success else 'pick_attempts_exhausted',
        'elapsed_seconds': 87.25,
        'detected_row': detected_row,
        'column_identified': True,
        'row_identified': detected_row is not None,
        'pick_attempts': 1,
        'navigation_goals': 4,
        'collision_episodes': 0,
        'bin_contact_confirmed': success,
        'target_book_model': 'book_col_3_row_3_blue' if success else None,
        'expected_target_book_model': 'book_col_3_row_3_blue',
        'target_identity_confirmed': success,
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _write_jsonl(path: Path, payloads) -> None:
    path.write_text(
        ''.join(json.dumps(payload, sort_keys=True) + '\n' for payload in payloads),
        encoding='utf-8',
    )


def test_five_complete_summaries_render_report_ready_csv_and_markdown(tmp_path):
    for index in range(1, 6):
        payload = _complete_summary(
            f'id-{index}',
            success=index != 4,
            detected_row=None if index == 4 else ((index - 1) % 4) + 1,
            row_identified=index != 4,
            elapsed_seconds=80.0 + index,
            failure_reason='' if index != 4 else 'trial_timeout',
            bin_contact_confirmed=index != 4,
        )
        _write_json(tmp_path / f'trial_{index}_summary.json', payload)

    report = summary.build_report([tmp_path])
    assert report.complete
    assert report.source_trial_count == 5
    assert len(report.rows) == 5
    assert all(row['status'] == 'COMPLETE' for row in report.rows)
    assert report.rows[3]['detected_row'] == summary.NOT_APPLICABLE

    csv_rows = list(csv.DictReader(io.StringIO(summary.render_csv(report))))
    assert len(csv_rows) == 5
    assert tuple(csv_rows[0]) == summary.TABLE_FIELDS
    assert csv_rows[0]['trial_id'] == 'id-1'
    assert csv_rows[3]['failure_reason'] == 'trial_timeout'
    assert csv_rows[4]['elapsed_seconds'] == '85.0'

    markdown = summary.render_markdown(report)
    assert 'Source-backed trials: **5/5**' in markdown
    assert 'Complete records: **5/5**' in markdown
    assert '| Trial 5 | id-5 |' in markdown


def test_summary_and_finished_jsonl_are_deduplicated_by_trial_id(tmp_path):
    payload = _complete_summary('same-id')
    _write_json(tmp_path / 'trial_same-id_summary.json', payload)
    _write_jsonl(
        tmp_path / 'trial_same-id.jsonl',
        [
            {
                'event': 'trial_started',
                'trial_id': 'same-id',
                'team_name': 'Test Team',
                'shelf_column_number': 3,
                'book_colour': 'blue',
                'wall_time_utc': '2026-08-27T12:00:00.000+00:00',
            },
            {
                'event': 'trial_finished',
                'wall_time_utc': '2026-08-27T12:02:00.000+00:00',
                **payload,
            },
        ],
    )

    report = summary.build_report([tmp_path])
    assert report.source_trial_count == 1
    assert report.rows[0]['status'] == 'COMPLETE'
    assert 'trial_same-id.jsonl' in report.rows[0]['source_files']
    assert 'trial_same-id_summary.json' in report.rows[0]['source_files']
    assert [row['status'] for row in report.rows[1:]] == [
        'MISSING TRIAL',
        'MISSING TRIAL',
        'MISSING TRIAL',
        'MISSING TRIAL',
    ]
    assert not report.complete


def test_partial_jsonl_never_infers_metrics_from_events(tmp_path):
    _write_jsonl(
        tmp_path / 'trial_partial.jsonl',
        [
            {
                'event': 'trial_started',
                'trial_id': 'partial',
                'team_name': 'Test Team',
                'shelf_column_number': 2,
                'book_colour': 'green',
            },
            {
                'event': 'navigation_goal',
                'trial_id': 'partial',
                'purpose': 'shelf_observation_pose',
            },
            {
                'event': 'unexpected_contact',
                'trial_id': 'partial',
                'count': 9,
            },
        ],
    )

    report = summary.build_report([tmp_path], expected_trials=1)
    row = report.rows[0]
    assert row['status'] == 'INCOMPLETE'
    assert row['trial_id'] == 'partial'
    assert row['navigation_goals'] == summary.MISSING
    assert row['collision_episodes'] == summary.MISSING
    assert row['elapsed_seconds'] == summary.MISSING
    assert 'JSONL has no trial_finished event' in row['issues']


def test_absent_field_is_missing_but_explicit_null_is_not_applicable(tmp_path):
    payload = _complete_summary(
        'incomplete',
        success=False,
        detected_row=None,
        row_identified=False,
        failure_reason='row_not_found',
        bin_contact_confirmed=False,
    )
    del payload['elapsed_seconds']
    _write_json(tmp_path / 'trial_incomplete_summary.json', payload)

    report = summary.build_report([tmp_path], expected_trials=1)
    row = report.rows[0]
    assert row['status'] == 'INCOMPLETE'
    assert row['elapsed_seconds'] == summary.MISSING
    assert row['detected_row'] == summary.NOT_APPLICABLE
    assert 'missing fields: elapsed_seconds' in row['issues']
    assert 'detected_row' not in row['issues']


def test_conflicting_summary_and_jsonl_values_are_not_silently_selected(tmp_path):
    json_summary = _complete_summary('conflict')
    jsonl_summary = dict(json_summary)
    jsonl_summary['success'] = False
    jsonl_summary['failure_reason'] = 'trial_timeout'
    jsonl_summary['bin_contact_confirmed'] = False
    _write_json(tmp_path / 'trial_conflict_summary.json', json_summary)
    _write_jsonl(
        tmp_path / 'trial_conflict.jsonl',
        [{'event': 'trial_finished', **jsonl_summary}],
    )

    report = summary.build_report([tmp_path], expected_trials=1)
    row = report.rows[0]
    assert row['status'] == 'INCOMPLETE'
    assert row['success'] == summary.CONFLICT
    assert row['failure_reason'] == summary.CONFLICT
    assert row['bin_contact_confirmed'] == summary.CONFLICT
    assert 'conflicting success: false versus true' in row['issues']


@pytest.mark.parametrize(
    ('field', 'bad_value', 'reason_fragment'),
    [
        ('shelf_column_number', 0, 'integer from 1 through 5'),
        ('success', 1, 'JSON boolean'),
        ('dry_run', 'false', 'JSON boolean'),
        ('target_identity_confirmed', 1, 'JSON boolean'),
        ('target_book_model', '', 'null or a non-empty model name'),
        ('expected_target_book_model', 3, 'null or a non-empty model name'),
        ('elapsed_seconds', -0.5, 'finite non-negative number'),
        ('detected_row', 5, 'null or an integer from 1 through 4'),
        ('navigation_goals', 2.5, 'non-negative integer'),
    ],
)
def test_invalid_values_remain_visible_and_are_flagged(
    tmp_path, field, bad_value, reason_fragment
):
    payload = _complete_summary('invalid', **{field: bad_value})
    _write_json(tmp_path / 'trial_invalid_summary.json', payload)

    report = summary.build_report([tmp_path], expected_trials=1)
    row = report.rows[0]
    assert row['status'] == 'INCOMPLETE'
    assert row[field].startswith(f'{summary.INVALID}:')
    assert reason_fragment in row['issues']


def test_malformed_and_empty_sources_are_visible_instead_of_disappearing(tmp_path):
    malformed = tmp_path / 'trial_corrupt_summary.json'
    malformed.write_text('{not json', encoding='utf-8')
    empty = tmp_path / 'trial_empty.jsonl'
    empty.write_text('', encoding='utf-8')

    report = summary.build_report([tmp_path], expected_trials=2)
    assert report.source_trial_count == 2
    assert all(row['status'] == 'INCOMPLETE' for row in report.rows)
    combined_issues = ' '.join(row['issues'] for row in report.rows)
    assert 'unreadable JSON' in combined_issues
    assert 'JSONL contains no event objects' in combined_issues
    assert all(row['trial_id'] == summary.MISSING for row in report.rows)


def test_missing_input_builds_five_explicit_missing_rows(tmp_path):
    missing_path = tmp_path / 'does-not-exist'
    report = summary.build_report([missing_path])
    assert report.source_trial_count == 0
    assert len(report.rows) == 5
    assert all(row['status'] == 'MISSING TRIAL' for row in report.rows)
    assert all(row['elapsed_seconds'] == summary.MISSING for row in report.rows)
    assert 'input path does not exist' in report.global_issues[0]
    assert '## Input warnings' in summary.render_markdown(report)


def test_more_than_expected_trials_is_an_error_not_silent_truncation(tmp_path):
    for index in range(6):
        _write_json(
            tmp_path / f'trial_{index}_summary.json',
            _complete_summary(f'extra-{index}'),
        )
    with pytest.raises(ValueError, match='found 6 distinct trial outputs'):
        summary.build_report([tmp_path])


def test_cli_writes_both_formats_and_returns_nonzero_for_incomplete_report(
    tmp_path, capsys
):
    results = tmp_path / 'results'
    results.mkdir()
    _write_json(results / 'trial_one_summary.json', _complete_summary('one'))
    csv_path = tmp_path / 'report' / 'trials.csv'
    markdown_path = tmp_path / 'report' / 'trials.md'

    exit_code = summary.main(
        [
            str(results),
            '--csv',
            str(csv_path),
            '--markdown',
            str(markdown_path),
        ]
    )
    assert exit_code == 2
    assert csv_path.is_file()
    assert markdown_path.is_file()
    assert len(list(csv.DictReader(io.StringIO(csv_path.read_text())))) == 5
    assert 'MISSING TRIAL' in markdown_path.read_text(encoding='utf-8')
    assert 'report is incomplete' in capsys.readouterr().err


def test_cli_returns_zero_only_for_five_complete_trials(tmp_path):
    for index in range(5):
        _write_json(
            tmp_path / f'trial_{index}_summary.json',
            _complete_summary(f'complete-{index}'),
        )
    assert summary.main([str(tmp_path), '--csv', str(tmp_path / 'table.csv')]) == 0


def test_dry_run_success_keeps_mode_and_identity_qualifiers_in_both_formats(tmp_path):
    payload = _complete_summary(
        'dry-run', dry_run=True, success=True, bin_contact_confirmed=False,
        target_book_model=None, expected_target_book_model=None,
        target_identity_confirmed=False,
    )
    _write_json(tmp_path / 'trial_dry_summary.json', payload)
    report = summary.build_report([tmp_path], expected_trials=1)
    assert report.complete  # A complete diagnostic record, not a delivered book.
    row = next(csv.DictReader(io.StringIO(summary.render_csv(report))))
    assert row['dry_run'] == row['success'] == 'true'
    assert row['bin_contact_confirmed'] == row['target_identity_confirmed'] == 'false'
    assert row['target_book_model'] == row['expected_target_book_model'] == summary.NOT_APPLICABLE
    markdown = summary.render_markdown(report)
    assert 'state-machine exercise, not physical delivery' in markdown
    assert 'record completeness, not mission success' in markdown


def test_legacy_success_without_mode_or_identity_remains_explicitly_incomplete(tmp_path):
    payload = _complete_summary('legacy')
    missing = ('dry_run', 'target_book_model', 'expected_target_book_model',
               'target_identity_confirmed')
    for field in missing:
        payload.pop(field)
    _write_json(tmp_path / 'trial_legacy_summary.json', payload)
    report = summary.build_report([tmp_path], expected_trials=1)
    assert not report.complete
    assert report.rows[0]['success'] == 'true'
    assert all(report.rows[0][field] == summary.MISSING for field in missing)


@pytest.mark.parametrize('overrides,invalid_field', [
    ({'bin_contact_confirmed': False}, 'bin_contact_confirmed'),
    ({'target_identity_confirmed': False}, 'target_identity_confirmed'),
    ({'target_book_model': None}, 'target_identity_confirmed'),
    ({'target_book_model': 'book_col_4_row_3_blue'}, 'target_identity_confirmed'),
])
def test_unqualified_physical_success_is_flagged_without_changing_source_result(
    tmp_path, overrides, invalid_field,
):
    _write_json(tmp_path / 'trial_inconsistent_summary.json',
                _complete_summary('inconsistent', **overrides))
    report = summary.build_report([tmp_path], expected_trials=1)
    assert not report.complete
    assert report.rows[0]['success'] == 'true'
    assert report.rows[0][invalid_field].startswith(summary.INVALID)


@pytest.mark.parametrize('field,other_value', [
    ('dry_run', True),
    ('target_book_model', 'book_col_4_row_3_blue'),
    ('target_identity_confirmed', False),
])
def test_mode_and_identity_conflicts_are_not_dropped_when_sources_merge(
    tmp_path, field, other_value,
):
    payload = _complete_summary('qualifier-conflict')
    _write_json(tmp_path / 'trial_conflict_summary.json', payload)
    _write_jsonl(tmp_path / 'trial_conflict.jsonl', [
        {'event': 'trial_finished', **payload, field: other_value},
    ])
    report = summary.build_report([tmp_path], expected_trials=1)
    assert not report.complete
    assert report.rows[0][field] == summary.CONFLICT
