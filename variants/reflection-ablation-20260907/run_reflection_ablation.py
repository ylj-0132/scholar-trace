"""Compare the supported Reflection inputs; historical four-group manifests are read-only evidence."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import MasterAction

SOURCE = ROOT / 'data/audits/harnessbank-20260906-routing-reconstructed-01/result.json'
GROUPS = ('full-history', 'rubric-union')
SCHEMA = 'reflection-context-comparison-v2'
RUBRICS = ('auxiliary_model_reliability', 'matched_resource_efficiency')
ANCHORS = ('r1-t3-f2', 'r1-t1-f3')
FOCUS = (
    'Using only the supplied Worker evidence, assess whether the reported model-role '
    'assignments and resource conditions justify treating the HarnessBank versus GEPA/DGM '
    'comparison as matched. Reconcile the reports about task agent, proposer/evolver and '
    'evaluator roles where possible. Distinguish explicit assignments, unresolved role '
    'equivalence, and genuinely missing information; identify any material correction or '
    'qualification and the smallest paper-internal check still needed, if any. '
    'Do not assume that a contradiction exists.'
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_state():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    calls = source['trace']['model_calls']
    state = copy.deepcopy(json.loads(next(c for c in calls if c['role'] == 'synthesis')['user_prompt'])['history'])
    previous = json.loads([c for c in calls if c['role'] == 'master'][-1]['user_prompt'])['state']
    state['history'] = [entry for entry in state['history'] if entry['kind'] != 'DECIDE']
    for key in ('provisional_assessment', 'unresolved_questions', 'checklist_coverage', 'remaining_rounds'):
        state[key] = copy.deepcopy(previous[key])
    state['cumulative_unresolved_questions'] = list(dict.fromkeys(
        question for entry in state['history'] for question in entry['unresolved_questions']))
    assert len(state['findings']) == len(previous['findings']) == 33
    assert len(state['reflection_reports']) == 1
    return state


def native_prompt(state, mode):
    captured = []
    def capture(**kwargs):
        captured.append(json.loads(kwargs['prompt']))
        return {'reflection_memo': 'offline capture only'}, None
    action = MasterAction('REFLECT', reflection_focus=FOCUS,
                         reflection_rubric_ids=RUBRICS, reflection_finding_ids=ANCHORS,
                         checklist_coverage=state['checklist_coverage'])
    shell = SimpleNamespace(steps=state['history'], remaining_rounds=state['remaining_rounds'])
    with patch.object(runtime, '_state_payload', return_value=copy.deepcopy(state)), patch.object(runtime, '_invoke_model', side_effect=capture):
        runtime.PaperReflector(llm=None, paper_name='harnessbank.pdf', page_index='[]',
                               overview_text='', paper_context_mode='master-overview-history-only',
                               reflection_context_mode=mode)(shell, trigger='master_requested',
                                   finding_ids=tuple(f['finding_id'] for f in state['findings']),
                                   proposed_decision=action)
    return captured[0]


def build_inputs(state):
    return {mode: native_prompt(state, mode) for mode in GROUPS}


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def provenance():
    from deep_research import experiment
    if Path(runtime.__file__).resolve().parent != ROOT / 'src/deep_research':
        raise ValueError('Comparison requires the active runtime, not a frozen variant')
    return {'source_result_sha256': digest(SOURCE), 'script_sha256': digest(__file__),
            'test_sha256': digest(Path(__file__).with_name('test_reflection_ablation.py')),
            'runtime_sha256': digest(runtime.__file__),
            'source_sha256': {name: digest(path) for name, path in experiment._reproducibility_sources().items()},
            'system_prompt_sha256': hashlib.sha256(runtime.REFLECTION_SYSTEM_PROMPT.encode()).hexdigest()}


def prepare(output):
    hashes = provenance()
    state = load_state()
    inputs = build_inputs(state)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'source_state.json', state)
    for group, payload in inputs.items():
        folder = output / group
        folder.mkdir()
        write_new(folder / 'prompt.json', payload)
    manifest = {'schema': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
                'source': SOURCE.relative_to(ROOT).as_posix() if SOURCE.is_relative_to(ROOT) else SOURCE.name, 'source_boundary': 'before final Master call; no final DECIDE or Synthesis answer',
                'hashes': hashes, 'groups': list(GROUPS), 'model': 'openai/gpt-5.6-terra',
                'temperature': 1.0, 'max_retries': 1, 'allow_json_repair': False, 'request_timeout': 90.0,
                'scope': 'current native full-history versus rubric-union inputs; not a replay of the historical four-group experiment',
                'focus': FOCUS, 'rubrics': list(RUBRICS), 'anchors': list(ANCHORS),
                'selection_limitation': 'One purposively selected role-comparability case; anchors intentionally do not enumerate all later relevant findings. Not a random sample.',
                'prompt_sha256': {g: digest(output / g / 'prompt.json') for g in GROUPS}}
    write_new(output / 'manifest.json', manifest)
    print(json.dumps({'prepared': True, 'findings': {g: len(p['state']['findings']) for g, p in inputs.items()}}), flush=True)


def run(output):
    from deep_research import experiment
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or manifest.get('groups') != list(GROUPS):
        raise ValueError('Unsupported or historical manifest; prepare a new two-policy comparison')
    if provenance() != manifest['hashes']:
        raise ValueError('Source or runner changed after preparation')
    if any((output / g / 'result.json').exists() for g in GROUPS) or (output / 'started.json').exists():
        raise ValueError('Create-once experiment already started; no automatic replay')
    for group in GROUPS:
        if digest(output / group / 'prompt.json') != manifest['prompt_sha256'][group]:
            raise ValueError('Prepared prompt changed')
    clients, keys = experiment._role_clients(None)
    client = clients['reflection']
    client.request_timeout = manifest['request_timeout']
    assert client.model == manifest['model'] and client.temperature == 1.0
    assert client.max_retries == 1 and not client.allow_json_repair
    write_new(output / 'started.json', {'at': datetime.now(timezone.utc).isoformat(), 'request_timeout': client.request_timeout})
    for group in GROUPS:
        folder = output / group
        def event(kind, record):
            safe, warnings = experiment._sanitize_audit_payload(asdict(record), ROOT, keys)
            with (folder / 'events.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'event': kind, 'record': safe, 'serialization_warnings': warnings}, ensure_ascii=False) + '\n')
        recorder = runtime.ModelCallRecorder(model=client.model, temperature=client.temperature, on_event=event)
        started = time.perf_counter()
        print(json.dumps({'group': group, 'status': 'started'}), flush=True)
        result = {'group': group, 'status': 'failed'}
        try:
            prompt_text = (folder / 'prompt.json').read_text(encoding='utf-8')
            prompt_context = json.loads(prompt_text)
            payload, _ = runtime._invoke_model(llm=client, prompt=prompt_text,
                                               system_prompt=runtime.REFLECTION_SYSTEM_PROMPT, recorder=recorder,
                                               role='reflection', round_number=3, task_question=FOCUS, image_inputs=())
            report = runtime.parse_reflection_report(payload, trigger='forced-context-comparison',
                                                     reflected_finding_ids=tuple(prompt_context['findings_to_reflect']))
            report = replace(report, context_mode=prompt_context['context_mode'],
                             context_finding_ids=tuple(f['finding_id'] for f in prompt_context['state']['findings']),
                             context_rubric_ids=tuple(prompt_context['context_rubric_ids']),
                             context_diagnostics=tuple(prompt_context['context_diagnostics']))
            result.update(status='completed', report=asdict(report))
        except Exception as exc:
            result['error'] = experiment._safe_error(str(exc), ROOT, keys)
        result.update(wall_seconds=round(time.perf_counter() - started, 3), calls=[asdict(r) for r in recorder.records])
        safe, warnings = experiment._sanitize_audit_payload(result, ROOT, keys)
        safe['serialization_warnings'] = warnings
        write_new(folder / 'result.json', safe)
        print(json.dumps({'group': group, 'status': result['status'], 'tokens': sum(r.total_tokens or 0 for r in recorder.records),
                          'wall_seconds': result['wall_seconds']}), flush=True)
    write_new(output / 'finished.json', {'at': datetime.now(timezone.utc).isoformat(), 'source_verified_after': provenance() == manifest['hashes']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True,
                        help='New create-once comparison directory; historical manifests cannot be run')
    args = parser.parse_args(argv)
    prepare(args.output) if args.operation == 'prepare' else run(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
