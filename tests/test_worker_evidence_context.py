import json
from pathlib import Path

from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import (
    AgentState, EvidenceTask, FindingEvidence, MasterAction, TraceStep,
    WorkerFinding, WorkerResult, _selected_research_context,
)
from deep_research.paper_reading import PaperPage


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete_json(self, prompt, **kwargs):
        self.prompts.append(json.loads(prompt))
        return next(self.responses)


def initial_state():
    task = EvidenceTask('Measure ablation', rubric_ids=('main_evidence',))
    result = WorkerResult(task, structured_findings=(WorkerFinding(
        'Removing graph lowers accuracy', (FindingEvidence('82 versus 90', 'text', 'p. 2 table'),),
        'Aggregate result only', content_rubric_ids=('ablation_or_counterevidence',)),))
    return AgentState(steps=(TraceStep(1, MasterAction('READ_PAPER', (task,)), (result,)),), findings=(result,))


def read(task, prior_ids):
    context, _ = _selected_research_context(initial_state(), task)
    llm = Responses({'page_ranges': [{'start': 3, 'end': 3}], 'rationale': 'Implementation'},
        {'findings': [{'finding': 'Removal changes access, qualifying r1-t1-f1',
            'evidence': [{'content': 'No entity expansion', 'evidence_type': 'text', 'locator': 'p. 3'}],
            'caveat': 'Does not isolate representation alone', 'prior_finding_ids': prior_ids}]})
    worker = runtime.PaperEvidenceWorker(pdf_path=Path('not-opened.pdf'),
        pages=[PaperPage(1, 'Introduction', 12, False, False), PaperPage(2, '82 versus 90', 12, False, False),
               PaperPage(3, 'No entity expansion', 20, False, False)], page_index='[]', llm=llm,
        render_pages=lambda *_: {3: 'data:image/png;base64,x'})
    worker.set_research_context(context, ())
    return worker(task), llm.prompts


def test_cross_rubric_dependency_survives_worker_and_downstream_history():
    task = EvidenceTask('Read removed access path', rubric_ids=('method_workflow',),
                        related_finding_ids=('r1-t1-f1',), decision_relevance='Look for complementary controls')
    result, prompts = read(task, ['r1-t1-f1'])
    assert result.error is None
    assert result.structured_findings[0].prior_finding_ids == ('r1-t1-f1',)
    for prompt in prompts:
        prior = prompt['research_context'][0]
        assert prior['evidence'] == '82 versus 90'
        assert prior['evidence_locator'] == 'p. 2 table'
        assert prior['caveat'] == 'Aggregate result only'
    state = initial_state()
    state = AgentState(steps=state.steps + (TraceStep(2, MasterAction('READ_PAPER', (task,)), (result,)),))
    for build in (runtime._state_payload, runtime._master_state_payload):
        assert build(state)['findings'][-1]['prior_finding_ids'] == ['r1-t1-f1']


def test_unknown_dependency_does_not_discard_current_page_finding():
    task = EvidenceTask('Read path', related_finding_ids=('r1-t1-f1',))
    result, _ = read(task, ['r1-t1-f1', 'r99-t1-f1'])
    assert result.error is None
    finding = result.structured_findings[0]
    assert finding.prior_finding_ids == ('r1-t1-f1',)
    assert any('unknown_prior_finding' in w for w in finding.content_warnings)
    assert finding.evidence[0].content == 'No entity expansion'


def test_independent_read_cannot_claim_historical_dependency():
    result, prompts = read(EvidenceTask('Read path independently', independent_read=True), ['r1-t1-f1'])
    assert result.error is None
    assert result.structured_findings[0].prior_finding_ids == ()
    assert all('research_context' not in p and 'decision_context' not in p for p in prompts)


def test_master_selects_complementary_content_without_requiring_contradiction():
    llm = Responses({'kind': 'NEEDS_HUMAN'})
    master = runtime.PaperAgentMaster(llm=llm, paper_name='paper', page_index='[]', overview_text='',
        overview_images=(), master_context_mode='incremental-with-evidence')
    master(initial_state())
    instructions = ' '.join(llm.prompts[0]['instructions'])
    assert 'complementary' in instructions
    assert 'content associations' in instructions
    assert 'preidentified contradiction' in instructions
