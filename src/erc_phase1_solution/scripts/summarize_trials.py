#!/usr/bin/env python3
"""Build source-backed CSV and Markdown tables from ERC trial JSON/JSONL files.

The mission manager writes both ``trial_<id>_summary.json`` and a JSONL event
stream whose final ``trial_finished`` event contains the same summary fields.
This script merges those representations by trial ID, reports conflicts, and
never computes or guesses a metric that was not explicitly recorded.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SUMMARY_FIELDS: Tuple[str, ...] = (
    'trial_id',
    'team_name',
    'shelf_column_number',
    'book_colour',
    'dry_run',
    'success',
    'failure_reason',
    'elapsed_seconds',
    'detected_row',
    'column_identified',
    'row_identified',
    'pick_attempts',
    'navigation_goals',
    'collision_episodes',
    'bin_contact_confirmed',
    'target_book_model',
    'expected_target_book_model',
    'target_identity_confirmed',
)

TABLE_FIELDS: Tuple[str, ...] = (
    'trial',
    *SUMMARY_FIELDS,
    'status',
    'issues',
    'source_files',
)

BOOLEAN_FIELDS: Set[str] = {
    'dry_run',
    'success',
    'column_identified',
    'row_identified',
    'bin_contact_confirmed',
    'target_identity_confirmed',
}
NONNEGATIVE_INTEGER_FIELDS: Set[str] = {
    'pick_attempts',
    'navigation_goals',
    'collision_episodes',
}
SUPPORTED_SUFFIXES = {'.json', '.jsonl'}
MISSING = 'MISSING'
NOT_APPLICABLE = 'N/A'
CONFLICT = 'CONFLICT'
INVALID = 'INVALID'


def _stable_value(value: Any) -> str:
    """Return a deterministic representation suitable for conflict messages."""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _same_value(first: Any, second: Any) -> bool:
    """Compare JSON values without treating booleans as integers."""

    if type(first) is not type(second):
        return False
    try:
        result = first == second
    except (TypeError, ValueError):
        return _stable_value(first) == _stable_value(second)
    if isinstance(result, bool):
        return result
    return _stable_value(first) == _stable_value(second)


@dataclass
class TrialRecord:
    """Explicit fields collected for one real trial output."""

    values: Dict[str, Any] = field(default_factory=dict)
    present: Set[str] = field(default_factory=set)
    conflicts: Dict[str, Set[str]] = field(default_factory=dict)
    sources: Set[Path] = field(default_factory=set)
    source_issues: List[str] = field(default_factory=list)
    sort_timestamp: Optional[str] = None

    def add_payload(self, payload: Mapping[str, Any], source: Path) -> None:
        """Merge only fields explicitly present in a JSON object."""

        self.sources.add(source)
        timestamp = payload.get('wall_time_utc')
        if isinstance(timestamp, str) and timestamp:
            if self.sort_timestamp is None or timestamp < self.sort_timestamp:
                self.sort_timestamp = timestamp
        for name in SUMMARY_FIELDS:
            if name not in payload:
                continue
            value = payload[name]
            if name not in self.present:
                self.values[name] = value
                self.present.add(name)
                continue
            if not _same_value(self.values[name], value):
                alternatives = self.conflicts.setdefault(
                    name, {_stable_value(self.values[name])}
                )
                alternatives.add(_stable_value(value))

    def merge(self, other: 'TrialRecord') -> None:
        """Merge another representation of the same trial."""

        for source in other.sources:
            self.sources.add(source)
        self.source_issues.extend(other.source_issues)
        if other.sort_timestamp is not None:
            if self.sort_timestamp is None or other.sort_timestamp < self.sort_timestamp:
                self.sort_timestamp = other.sort_timestamp
        for name in other.present:
            self.add_payload({name: other.values[name]}, min(other.sources))
        for name, alternatives in other.conflicts.items():
            target = self.conflicts.setdefault(name, set())
            if name in self.present:
                target.add(_stable_value(self.values[name]))
            target.update(alternatives)


@dataclass(frozen=True)
class Validation:
    """Completeness and validity details for one record."""

    missing: Tuple[str, ...]
    invalid: Mapping[str, str]
    conflicts: Mapping[str, Set[str]]
    issues: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether every metric is present, valid, and unambiguous."""

        return not (self.missing or self.invalid or self.conflicts or self.issues)


@dataclass(frozen=True)
class TrialReport:
    """Rendered rows plus integrity metadata."""

    rows: Tuple[Mapping[str, str], ...]
    source_trial_count: int
    expected_trial_count: int
    global_issues: Tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether exactly the expected complete trials were found."""

        return (
            self.source_trial_count == self.expected_trial_count
            and not self.global_issues
            and all(row['status'] == 'COMPLETE' for row in self.rows)
        )


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_field(name: str, value: Any) -> Optional[str]:
    if name == 'trial_id':
        if not isinstance(value, str) or not value.strip():
            return 'must be a non-empty string'
    elif name == 'team_name':
        if not isinstance(value, str) or not value.strip():
            return 'must be a non-empty string'
    elif name == 'shelf_column_number':
        if not _is_integer(value) or not 1 <= value <= 5:
            return 'must be an integer from 1 through 5'
    elif name == 'book_colour':
        if not isinstance(value, str) or value.lower() not in {
            'red', 'blue', 'green', 'yellow'
        }:
            return 'must be red, blue, green, or yellow'
    elif name in BOOLEAN_FIELDS:
        if not isinstance(value, bool):
            return 'must be a JSON boolean'
    elif name == 'failure_reason':
        if not isinstance(value, str):
            return 'must be a string (empty is valid for a successful trial)'
    elif name in ('target_book_model', 'expected_target_book_model'):
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            return 'must be null or a non-empty model name'
    elif name == 'elapsed_seconds':
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            return 'must be a finite non-negative number'
    elif name == 'detected_row':
        if value is not None and (not _is_integer(value) or not 1 <= value <= 4):
            return 'must be null or an integer from 1 through 4'
    elif name in NONNEGATIVE_INTEGER_FIELDS:
        if not _is_integer(value) or value < 0:
            return 'must be a non-negative integer'
    return None


def validate_record(record: TrialRecord) -> Validation:
    """Validate schema without filling absent or null metrics."""

    missing = tuple(name for name in SUMMARY_FIELDS if name not in record.present)
    invalid: Dict[str, str] = {}
    for name in SUMMARY_FIELDS:
        if name not in record.present or name in record.conflicts:
            continue
        reason = _validate_field(name, record.values[name])
        if reason is not None:
            invalid[name] = reason

    if not invalid and not record.conflicts:
        success = record.values.get('success')
        failure_reason = record.values.get('failure_reason')
        if success is False and failure_reason == '':
            invalid['failure_reason'] = 'a failed trial must record its reason'
        if success is True and isinstance(failure_reason, str) and failure_reason:
            invalid['failure_reason'] = 'a successful trial must have an empty reason'
        if record.values.get('row_identified') is True:
            if record.values.get('detected_row') is None:
                invalid['detected_row'] = 'cannot be null when row_identified is true'
        if record.values.get('target_identity_confirmed') is True:
            target = record.values.get('target_book_model')
            expected = record.values.get('expected_target_book_model')
            if not target or not expected or target != expected:
                invalid['target_identity_confirmed'] = (
                    'confirmed identity requires matching target and expected model names'
                )
        if success is True and record.values.get('dry_run') is False:
            if record.values.get('bin_contact_confirmed') is not True:
                invalid['bin_contact_confirmed'] = (
                    'a non-dry-run success must record target/bin contact'
                )
            if record.values.get('target_identity_confirmed') is not True:
                invalid['target_identity_confirmed'] = (
                    'a non-dry-run success must record confirmed target identity'
                )

    return Validation(
        missing=missing,
        invalid=invalid,
        conflicts=record.conflicts,
        issues=tuple(dict.fromkeys(record.source_issues)),
    )


def _record_identity(record: TrialRecord) -> Optional[str]:
    if 'trial_id' not in record.present or 'trial_id' in record.conflicts:
        return None
    value = record.values.get('trial_id')
    return value if isinstance(value, str) and value.strip() else None


def _parse_json(path: Path) -> List[TrialRecord]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        record = TrialRecord(sources={path})
        record.source_issues.append(f'unreadable JSON: {error}')
        return [record]

    if isinstance(raw, dict) and isinstance(raw.get('trials'), list):
        objects: Iterable[Any] = raw['trials']
    elif isinstance(raw, list):
        objects = raw
    else:
        objects = (raw,)

    records: List[TrialRecord] = []
    for index, payload in enumerate(objects, start=1):
        record = TrialRecord(sources={path})
        if isinstance(payload, dict):
            record.add_payload(payload, path)
        else:
            record.source_issues.append(
                f'JSON item {index} is {_stable_value(payload)}, not an object'
            )
        records.append(record)
    if not records:
        record = TrialRecord(sources={path})
        record.source_issues.append('JSON contains no trial objects')
        records.append(record)
    return records


def _parse_jsonl(path: Path) -> List[TrialRecord]:
    grouped: Dict[Tuple[str, str], TrialRecord] = {}
    grouped_events: Dict[Tuple[str, str], List[Any]] = {}
    parse_issues: List[str] = []
    valid_objects = 0
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        record = TrialRecord(sources={path})
        record.source_issues.append(f'unreadable JSONL: {error}')
        return [record]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            parse_issues.append(f'line {line_number} is invalid JSON: {error.msg}')
            continue
        if not isinstance(payload, dict):
            parse_issues.append(f'line {line_number} is not a JSON object')
            continue
        valid_objects += 1
        raw_id = payload.get('trial_id')
        if isinstance(raw_id, str) and raw_id.strip():
            key = ('trial_id', raw_id)
        else:
            key = ('missing_id', str(path))
        record = grouped.setdefault(key, TrialRecord(sources={path}))
        grouped_events.setdefault(key, []).append(payload.get('event'))
        record.add_payload(payload, path)

    if not grouped:
        record = TrialRecord(sources={path})
        if valid_objects == 0 and not parse_issues:
            record.source_issues.append('JSONL contains no event objects')
        grouped[('empty', str(path))] = record

    for key, record in grouped.items():
        record.source_issues.extend(parse_issues)
        # A JSONL stream without the final summary is a genuine partial trial;
        # do not infer counts from commands or contact events.
        if 'trial_finished' not in grouped_events.get(key, []):
            record.source_issues.append('JSONL has no trial_finished event')
    return list(grouped.values())


def parse_trial_file(path: Path) -> List[TrialRecord]:
    """Parse one JSON or JSONL file into one or more explicit trial records."""

    suffix = path.suffix.lower()
    if suffix == '.json':
        return _parse_json(path)
    if suffix == '.jsonl':
        return _parse_jsonl(path)
    raise ValueError(f'unsupported trial file extension: {path}')


def discover_trial_files(inputs: Sequence[Path]) -> Tuple[List[Path], List[str]]:
    """Resolve explicit files and recursively find ``trial*.json[l]`` in dirs."""

    discovered: Set[Path] = set()
    issues: List[str] = []
    for raw_path in inputs:
        path = raw_path.expanduser()
        if not path.exists():
            issues.append(f'input path does not exist: {path}')
            continue
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                discovered.add(path.resolve())
            else:
                issues.append(f'unsupported input file: {path}')
            continue
        matches = [
            candidate.resolve()
            for candidate in path.rglob('trial*')
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not matches:
            issues.append(f'no trial JSON/JSONL files found in: {path}')
        discovered.update(matches)
    return sorted(discovered, key=lambda candidate: str(candidate).lower()), issues


def load_trial_records(paths: Sequence[Path]) -> List[TrialRecord]:
    """Parse files and deduplicate summary/event representations by trial ID."""

    parsed: List[TrialRecord] = []
    for path in paths:
        parsed.extend(parse_trial_file(path))

    identified: Dict[str, TrialRecord] = {}
    unidentified: List[TrialRecord] = []
    for record in parsed:
        identity = _record_identity(record)
        if identity is None:
            unidentified.append(record)
        elif identity in identified:
            identified[identity].merge(record)
        else:
            identified[identity] = record
    return list(identified.values()) + unidentified


def _record_sort_key(record: TrialRecord) -> Tuple[str, str, str]:
    timestamp = record.sort_timestamp or '\uffff'
    identity = _record_identity(record) or '\uffff'
    sources = ';'.join(str(path).lower() for path in sorted(record.sources))
    return timestamp, identity, sources


def _display_value(record: TrialRecord, validation: Validation, name: str) -> str:
    if name in validation.conflicts:
        return CONFLICT
    if name not in record.present:
        return MISSING
    value = record.values[name]
    if name in validation.invalid:
        return f'{INVALID}: {_stable_value(value)}'
    if value is None:
        return NOT_APPLICABLE
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _issue_text(validation: Validation) -> str:
    details: List[str] = []
    if validation.missing:
        details.append('missing fields: ' + ', '.join(validation.missing))
    for name, alternatives in sorted(validation.conflicts.items()):
        details.append(
            f'conflicting {name}: ' + ' versus '.join(sorted(alternatives))
        )
    for name, reason in sorted(validation.invalid.items()):
        details.append(f'invalid {name}: {reason}')
    details.extend(validation.issues)
    return '; '.join(details)


def _record_row(record: TrialRecord, slot: int) -> Dict[str, str]:
    validation = validate_record(record)
    row = {'trial': f'Trial {slot}'}
    for name in SUMMARY_FIELDS:
        row[name] = _display_value(record, validation, name)
    row['status'] = 'COMPLETE' if validation.complete else 'INCOMPLETE'
    row['issues'] = _issue_text(validation)
    row['source_files'] = '; '.join(str(path) for path in sorted(record.sources))
    return row


def _missing_row(slot: int) -> Dict[str, str]:
    row = {'trial': f'Trial {slot}'}
    for name in SUMMARY_FIELDS:
        row[name] = MISSING
    row['status'] = 'MISSING TRIAL'
    row['issues'] = 'No real trial output was found for this required slot'
    row['source_files'] = MISSING
    return row


def build_report(
    inputs: Sequence[Path],
    *,
    expected_trials: int = 5,
) -> TrialReport:
    """Build exactly ``expected_trials`` rows without inventing absent data."""

    if expected_trials < 1:
        raise ValueError('expected_trials must be at least 1')
    files, discovery_issues = discover_trial_files(inputs)
    records = sorted(load_trial_records(files), key=_record_sort_key)
    if len(records) > expected_trials:
        raise ValueError(
            f'found {len(records)} distinct trial outputs but expected {expected_trials}; '
            'use a directory containing only the intended report trials'
        )
    rows: List[Mapping[str, str]] = [
        _record_row(record, index)
        for index, record in enumerate(records, start=1)
    ]
    for slot in range(len(rows) + 1, expected_trials + 1):
        rows.append(_missing_row(slot))
    return TrialReport(
        rows=tuple(rows),
        source_trial_count=len(records),
        expected_trial_count=expected_trials,
        global_issues=tuple(discovery_issues),
    )


def render_csv(report: TrialReport) -> str:
    """Render report rows as a conventional CSV table."""

    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(report.rows)
    return stream.getvalue()


def _markdown_cell(value: str) -> str:
    return value.replace('\\', '\\\\').replace('|', '\\|').replace('\r', '').replace(
        '\n', '<br>'
    )


def render_markdown(report: TrialReport) -> str:
    """Render a report-ready Markdown table with an integrity summary."""

    complete_count = sum(row['status'] == 'COMPLETE' for row in report.rows)
    lines = [
        '# ERC Phase 1 Trial Results',
        '',
        (
            f'Source-backed trials: **{report.source_trial_count}/'
            f'{report.expected_trial_count}**. Complete records: '
            f'**{complete_count}/{report.expected_trial_count}**.'
        ),
        '',
        (
            '> Integrity rule: `MISSING` means the field or required trial was not '
            'present in the supplied files; `N/A` means the source explicitly recorded null.'
        ),
        '',
        (
            '> Result scope: `success` is the source-reported mission result. '
            '`dry_run=true` is a state-machine exercise, not physical delivery. '
            '`dry_run=MISSING` has unknown mode. Identity and bin-contact fields '
            'do not alone prove post-release placement inside the bin. '
            '`COMPLETE` describes record completeness, not mission success. '
            '`elapsed_seconds` retains its source clock and endpoints; this '
            'table does not convert it to launch-to-contact time.'
        ),
        '',
        '| ' + ' | '.join(TABLE_FIELDS) + ' |',
        '| ' + ' | '.join('---' for _ in TABLE_FIELDS) + ' |',
    ]
    for row in report.rows:
        lines.append(
            '| '
            + ' | '.join(_markdown_cell(str(row[name])) for name in TABLE_FIELDS)
            + ' |'
        )
    if report.global_issues:
        lines.extend(('', '## Input warnings', ''))
        lines.extend(f'- {_markdown_cell(issue)}' for issue in report.global_issues)
    return '\n'.join(lines) + '\n'


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline='')


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate source-backed ERC trial JSON/JSONL into a five-trial '
            'CSV and Markdown report table.'
        )
    )
    parser.add_argument(
        'inputs',
        nargs='*',
        type=Path,
        default=[Path('results')],
        help='trial JSON/JSONL files or result directories (default: results)',
    )
    parser.add_argument(
        '--expected-trials',
        type=int,
        default=5,
        help='required number of report rows (default: 5)',
    )
    parser.add_argument('--csv', type=Path, help='write the CSV table to this path')
    parser.add_argument(
        '--markdown', type=Path, help='write the Markdown table to this path'
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line summarizer and return a process exit code."""

    arguments = _argument_parser().parse_args(argv)
    try:
        report = build_report(
            arguments.inputs, expected_trials=arguments.expected_trials
        )
    except ValueError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    csv_text = render_csv(report)
    markdown_text = render_markdown(report)
    if arguments.csv is not None:
        _write_output(arguments.csv, csv_text)
    if arguments.markdown is not None:
        _write_output(arguments.markdown, markdown_text)
    if arguments.csv is None and arguments.markdown is None:
        print(markdown_text, end='')
    for issue in report.global_issues:
        print(f'WARNING: {issue}', file=sys.stderr)
    if not report.complete:
        print(
            'WARNING: report is incomplete; inspect status and issues columns',
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
