import pytest
from backend.grounding import render_grounded_answer, resolve_source_scope


def test_nonconsecutive_sources_are_numbered_by_first_use():
    evidence = {'E1': {'id': 'a'}, 'E5': {'id': 'b'}}
    status, text, citations = render_grounded_answer({'status': 'answered', 'claims': [
        {'text': 'An experience claim [5].', 'evidence_ids': ['E5']},
        {'text': 'Another claim.', 'evidence_ids': ['E1', 'E5', 'E1']},
    ]}, evidence)
    assert status == 'answered'
    assert citations == [evidence['E5'], evidence['E1']]
    assert '[5]' not in text
    assert 'Another claim. [2] [1]' in text


@pytest.mark.parametrize('refs', [['E99'], [], [7], 'E1'])
def test_unsupported_references_do_not_block_answer_or_invent_sources(refs):
    status, text, citations = render_grounded_answer({'status': 'answered', 'claims': [{'text': 'Claim', 'evidence_ids': refs}]}, {'E1': {'id': 'a'}})
    assert status == 'partially_answered'
    assert text.startswith('Claim')
    assert 'could not be verified' in text
    assert citations == []


def test_abstention_has_no_sources_or_model_invented_facts():
    assert render_grounded_answer({'status': 'no_evidence', 'claims': []}, {})[2] == []


def test_named_resume_question_excludes_other_resume_and_media():
    assets = [
        {'id':'ai', 'name':'Sajjad_AI_ML_Engineer.pdf', 'kind':'pdf', 'status':'ready'},
        {'id':'cv', 'name':'Computer_Vision_Resume.pdf', 'kind':'pdf', 'status':'ready'},
        {'id':'v', 'name':'AI_ML_video.mp4', 'kind':'video', 'status':'ready'},
    ]
    assert resolve_source_scope('what my AI ML resume says about experience', None, assets) == ['ai']
    assert resolve_source_scope('what my AI ML resume says', ['cv'], assets) == ['cv']


def test_named_video_question_excludes_other_sources():
    assets = [
        {'id':'sample', 'name':'sample_video.mp4', 'kind':'video', 'status':'ready'},
        {'id':'demo', 'name':'Demo shapes video.mp4', 'kind':'video', 'status':'ready'},
        {'id':'cv', 'name':'Resume.pdf', 'kind':'pdf', 'status':'ready'},
    ]
    assert resolve_source_scope('what is in the sample video?', None, assets) == ['sample']


def test_plain_answer_format_does_not_block_useful_output():
    status, text, citations = render_grounded_answer({'status':'answered', 'answer':'A visible circle.'}, {'E1':{'id':'a'}})
    assert status == 'partially_answered'
    assert 'A visible circle.' in text
    assert citations == []


def test_missing_answer_format_returns_labeled_retrieved_material():
    status, text, citations = render_grounded_answer({'status':'answered'}, {'E1':{'id':'a', 'text':'A blue circle.', 'seconds':0}})
    assert status == 'partially_answered'
    assert 'retrieved passages' in text and '0:00' in text and '[1]' in text
    assert citations[0]['id'] == 'a'
