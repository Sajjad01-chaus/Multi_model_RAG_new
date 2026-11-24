"""Acceptance-test explicitly supplied local files; reports stay in ignored data/."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    args = parser.parse_args()
    report = []
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=180) as client:
        for path, kind, question in [(args.pdf, 'pdf', 'What technical skills and computer vision experience are stated in this resume? Summarize only supported details.'), (args.video, 'video', 'Describe the main visible content of this video and include the capture timestamps from the source metadata. Do not infer speech or unseen events.')]:
            filename = path.name if path.suffix else path.name + '.pdf'
            with path.open('rb') as source:
                response = await client.post('/api/assets', files={'file': (filename, source)})
            response.raise_for_status()
            aid = response.json()['id']
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                assets = (await client.get('/api/assets')).json()
                asset = next(a for a in assets if a['id'] == aid)
                if asset['status'] in ('ready', 'failed'):
                    break
                await asyncio.sleep(2)
            print(kind + ' ingestion: ' + asset['status'], flush=True)
            if asset['status'] != 'ready':
                raise RuntimeError(asset.get('error') or 'Ingestion timed out')
            response = await client.post('/api/chat', json={'question': question, 'asset_ids': [aid], 'detail': 'brief'})
            response.raise_for_status()
            answer = response.json()
            passed = answer['status'] in ('answered', 'partially_answered') and bool(answer['citations']) and all(c['asset_id'] == aid for c in answer['citations'])
            print(kind + ' grounded answer and scoped citations: ' + ('PASS' if passed else 'FAIL'), flush=True)
            if not passed:
                print('Answer status: ' + answer['status'], flush=True)
            for c in answer['citations']:
                source = await client.get(c['source_url'].split('#')[0])
                assert source.status_code == 200, 'Source link failed'
            if kind == 'video':
                passed = passed and all(c.get('seconds') is not None for c in answer['citations'])
                print('Video timestamp citations: ' + ('PASS' if passed else 'FAIL'), flush=True)
            report.append({'kind': kind, 'asset_id': aid, 'passed': passed, 'answer': answer})
    destination = Path(__file__).resolve().parents[1] / 'data' / 'user-acceptance.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if not all(r['passed'] for r in report):
        raise SystemExit('Some acceptance checks failed; see local data/user-acceptance.json')
    print('User-file acceptance checks passed. Detailed answers remain in local data/user-acceptance.json.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
