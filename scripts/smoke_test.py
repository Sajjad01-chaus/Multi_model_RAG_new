"""Exercise real HTTP, Groq and MCP paths. Uses only synthetic test content."""
import asyncio
import json
import re
from pathlib import Path
import subprocess
import time

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'data' / 'test-fixtures'
FIXTURES.mkdir(parents=True, exist_ok=True)
BASE = 'http://127.0.0.1:8000'
checks = []


def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    print(('PASS ' if condition else 'FAIL ') + name, flush=True)
    if not condition:
        raise AssertionError(name)


def fixtures():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 16 Tf 50 730 Td (Prism Demo Project Handbook) Tj 0 -30 Td (Project AURORA-742 launches on 15 December 2026.) Tj 0 -25 Td (The project budget is 42000 rupees.) Tj 0 -25 Td (Mira Patel is the project manager.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.write(FIXTURES / 'Demo project handbook.pdf')
    image = Image.new('RGB', (900, 540), '#f7f8fc')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 36)
    except OSError:
        font = ImageFont.load_default(size=36)
    draw.text((50, 35), 'PRISM DEMO: VISUAL EVIDENCE', fill='#333344', font=font)
    draw.rectangle((70, 160, 330, 410), fill='#dd4035')
    draw.ellipse((500, 160, 760, 420), fill='#3277dd')
    draw.text((70, 455), 'Red square', fill='#333344', font=font)
    draw.text((500, 455), 'Blue circle', fill='#333344', font=font)
    image.save(FIXTURES / 'Demo shapes.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', str(FIXTURES / 'Demo shapes.png'), '-t', '2', '-r', '5', '-pix_fmt', 'yuv420p', str(FIXTURES / 'Demo shapes video.mp4')], check=True, capture_output=True)


async def main():
    fixtures()
    ids = []
    async with httpx.AsyncClient(base_url=BASE, timeout=180) as client:
        health = (await client.get('/api/health')).json()
        check('Backend ready with configured Groq key', health['status'] == 'ready' and health['provider_configured'])
        bad = await client.post('/api/assets', files={'file': ('bad.pdf', b'not a pdf')})
        check('Invalid PDF rejected', bad.status_code == 400)
        for path in [FIXTURES / 'Demo project handbook.pdf', FIXTURES / 'Demo shapes.png', FIXTURES / 'Demo shapes video.mp4']:
            existing = (await client.get('/api/assets')).json()
            same = [a for a in existing if a['name'] == path.name and a['status'] == 'ready']
            if same:
                ids.append(same[0]['id'])
                print('Reusing synthetic fixture: ' + path.suffix, flush=True)
                continue
            for old in existing:
                if old['name'] == path.name and old['status'] == 'failed':
                    await client.delete('/api/assets/' + old['id'])
            with path.open('rb') as f:
                response = await client.post('/api/assets', files={'file': (path.name, f)})
            check(f'Upload accepted: {path.suffix}', response.status_code == 202)
            aid = response.json()['id']
            ids.append(aid)
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                assets = (await client.get('/api/assets')).json()
                asset = next(a for a in assets if a['id'] == aid)
                if asset['status'] in ('ready', 'failed'):
                    break
                await asyncio.sleep(1)
            if asset.get('error'):
                print('Ingestion error:', asset['error'])
            check(f'Extraction and indexing: {path.suffix}', asset['status'] == 'ready')

        async with streamablehttp_client(BASE + '/mcp/') as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                check('MCP discovers all three tools', {t.name for t in tools.tools} == {'search_documents', 'analyze_image', 'retrieve_video_frames'})

                async def call(name, args):
                    result = await session.call_tool(name, args)
                    check('MCP call ' + name, not result.isError)
                    return result.structuredContent or json.loads(next(c.text for c in result.content if c.type == 'text'))

                result = await call('search_documents', {'query': 'AURORA-742 budget', 'asset_ids': [ids[0]]})
                check('Both hybrid retrieval channels return candidates', result['channels']['dense'] > 0 and result['channels']['keyword'] > 0)
                check('Source scope enforced before fusion', all(r['asset_id'] == ids[0] for r in result['results']))
                result = await call('search_documents', {'query': 'How much money was allocated to the initiative?', 'asset_ids': [ids[0]]})
                check('Semantic paraphrase retrieves budget evidence', any('42000' in r['text'] for r in result['results']))
                result = await call('search_documents', {'query': '" OR * ( AND NOT :', 'asset_ids': [ids[0]]})
                check('FTS query syntax safely escaped', isinstance(result['results'], list))
                frames = await call('retrieve_video_frames', {'video_id': ids[2], 'query': 'blue circle', 'max_frames': 3})
                check('Video tool returns timestamped frames', bool(frames['frames']) and frames['frames'][0]['seconds'] == 0)
                image = await call('search_documents', {'query': 'red square blue circle', 'asset_ids': [ids[1]]})
                analyzed = await call('analyze_image', {'artifact_id': image['results'][0]['id'], 'question': 'What color is the circle?'})
                check('Original image inspection recognizes blue circle', 'blue' in analyzed['observations'].lower())

        for name, question, scope, term in [
            ('PDF grounded answer', 'What is the project budget?', [ids[0]], '42'),
            ('Image grounded answer', 'What color is the circle?', [ids[1]], 'blue'),
            ('Video grounded answer', 'What shapes are visible in the video and at what timestamp?', [ids[2]], 'circle'),
            ('No-evidence abstention', 'What is the secret password for the Mars colony?', [ids[0]], None),
        ]:
            response = await client.post('/api/chat', json={'question': question, 'asset_ids': scope, 'detail': 'brief', 'language': 'English'})
            answer = response.json()
            print(name + ': ' + answer['answer'][:280], flush=True)
            check(name, answer['status'] == 'no_evidence' if term is None else answer['status'] in ('answered', 'partially_answered') and term in answer['answer'].lower() and bool(answer['citations']))
            if term:
                check('Human-readable citation numbers', '[1]' in answer['answer'] and not re.search(r'\[[0-9a-f]{8}-', answer['answer']))
                for citation in answer['citations']:
                    source = await client.get(citation['source_url'].split('#')[0])
                    check('Citation source resolves', source.status_code == 200)
                check('Chat executed real MCP search', any(t['tool'] == 'search_documents' and t['ok'] for t in answer['trace']))
                if scope == [ids[2]]:
                    check('Video chat executes all three MCP tools', len({t['tool'] for t in answer['trace']}) == 3)
                    check('Video answer uses capture timestamp', '0:00' in answer['answer'])
        video = await client.get(f'/api/assets/{ids[2]}/content', headers={'Range': 'bytes=0-99'})
        check('Video content supports range requests', video.status_code == 206 and len(video.content) == 100)
        # Clean our synthetic assets only; never delete a user's own files.
        for aid in ids:
            response = await client.delete('/api/assets/' + aid)
            check('Synthetic source cleanup', response.status_code == 200)
        result = await client.post('/api/chat', json={'question': 'budget', 'asset_ids': ids})
        check('Deleted assets excluded from search', result.json()['status'] == 'no_evidence')
    (FIXTURES / 'smoke-results.json').write_text(json.dumps(checks, indent=2))
    print(f'Completed {len(checks)} checks.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
