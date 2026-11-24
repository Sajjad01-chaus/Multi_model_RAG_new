"""Render verified claim references into deterministic source numbers."""
import re


def render_grounded_answer(payload: dict, evidence: dict) -> tuple[str, str, list]:
    status = payload.get('status')
    if status not in ('answered', 'partially_answered', 'no_evidence'):
        raise ValueError('The model returned an invalid answer status. Please retry.')
    if status == 'no_evidence':
        return status, "I couldn't find supporting information for this question in the selected files.", []
    claims = payload.get('claims')
    if isinstance(claims, dict):
        claims = [claims]
    if not isinstance(claims, list) or not claims:
        plain = payload.get('answer')
        if isinstance(plain, str) and plain.strip():
            claims = [{'text': plain, 'evidence_ids': payload.get('evidence_ids', [])}]
        else:
            # A response-format failure can still show actual retrieved material.
            citations = list(evidence.values())[:3]
            excerpts = []
            for index, item in enumerate(citations, 1):
                locator = f"Page {item['page']}" if item.get('page') else ''
                if item.get('seconds') is not None:
                    seconds = item['seconds']
                    locator = f'{int(seconds // 60)}:{int(seconds % 60):02d}'
                excerpts.append(f"{locator}: {item.get('text', '')} [{index}]".strip(': '))
            return 'partially_answered', 'I found relevant source material, but could not format a complete answer. Here are the retrieved passages:\n\n' + '\n\n'.join(excerpts), citations
    claims = claims[:40]
    citations, numbers, paragraphs = [], {}, []
    missing_references = False
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError('The model returned an invalid claim. Please retry.')
        text, refs = claim.get('text') or claim.get('claim') or claim.get('content'), claim.get('evidence_ids')
        if not isinstance(text, str) or not text.strip():
            continue
        supplied = refs if isinstance(refs, list) else []
        refs = [ref for ref in supplied if isinstance(ref, str) and ref in evidence]
        if not refs or len(refs) != len(supplied):
            missing_references = True
        # Source numbering belongs to the application, never the language model.
        text = re.sub(r'\[(?:\d+(?:\s*,\s*\d+)*|E\d+)\]', '', text).strip()
        for ref in dict.fromkeys(refs):
            if ref not in numbers:
                numbers[ref] = len(citations) + 1
                citations.append(evidence[ref])
        labels = ' '.join(f'[{numbers[ref]}]' for ref in dict.fromkeys(refs))
        paragraphs.append(f'{text} {labels}'.strip())
    if not paragraphs:
        raise ValueError('The model returned an empty answer. Please retry.')
    if missing_references:
        status = 'partially_answered'
        paragraphs.append('Some source references could not be verified. Check those details against the original files.')
    elif status == 'partially_answered':
        paragraphs.append('The selected sources do not fully answer every part of the question.')
    return status, '\n\n'.join(paragraphs), citations


def resolve_source_scope(question: str, explicit: list[str] | None, assets: list[dict]) -> list[str] | None:
    """Honor explicit selection; narrow named resume requests before retrieval."""
    if explicit is not None:
        return explicit
    if re.search(r'\bvideo\b', question, re.I):
        videos = [a for a in assets if a['status'] == 'ready' and a['kind'] == 'video']
        tokens = set(re.findall(r'[a-z0-9]+', question.lower())) - {'what','is','in','the','video','a','an','of','this','my','show','me','about'}
        matched = [a for a in videos if tokens & set(re.findall(r'[a-z0-9]+', a['name'].lower()))]
        return [a['id'] for a in matched or videos] if videos else None
    if not re.search(r'\b(resume|cv)\b', question, re.I):
        return None
    documents = [a for a in assets if a['status'] == 'ready' and a['kind'] in ('pdf', 'text')]
    words = set(re.findall(r'[a-z0-9]+', question.lower()))
    specific = words & {'ai', 'ml', 'computer', 'vision'}
    matched = [a for a in documents if specific and specific.issubset(set(re.findall(r'[a-z0-9]+', a['name'].lower()))) ]
    scope = matched or documents
    return [a['id'] for a in scope] if scope else None
