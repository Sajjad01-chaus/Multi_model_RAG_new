# Prism Multimodal RAG Assistant

A local React + FastAPI research workspace for questions over PDFs, text, images, and short videos. Answers use retrieved evidence and link to the original page, image, or video timestamp.

## Run locally

Requires Python 3.12+, Node.js 20.19+ (or 22.12+), and FFmpeg/ffprobe on PATH. Keep this service on localhost: this demo has no authentication.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY. Do not overwrite an existing configured .env.
cd frontend
npm install
npm run build
cd ..
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open [the local app](http://127.0.0.1:8000/). The built React app is served by FastAPI. In this prepared workspace, dependencies and frontend build already exist: run only the final command, or `./start.ps1`. The first startup downloads approximately 90 MB of MiniLM ONNX weights; later starts use the local cache. No PyTorch, GPU, hosted vector database, or paid API is needed.

For frontend development, run `npm run dev` in `frontend` and open [the Vite development server](http://127.0.0.1:5173/). Vite proxies `/api` to the backend. Restart the backend after a production frontend build so the static mount is available. Use one backend worker; the embedded vector store must have one owner.

## Features

- PDF page extraction and TXT/pasted text ingestion.
- Groq image understanding and up to six sampled video frames with exact timestamps.
- Hybrid RAG: local MiniLM semantic vectors in Qdrant + SQLite FTS5 BM25 keyword search, combined with reciprocal rank fusion.
- Selected-source filtering in both retrieval channels before ranking fusion.
- Groq grounded answers, validated citation IDs, and no-evidence/temporary-unavailability responses.
- Answer length and language preferences, source previews, video seeking, and ingestion status.
- Real MCP server at `http://127.0.0.1:8000/mcp/` with `search_documents`, `analyze_image`, and `retrieve_video_frames`. Chat connects with an MCP client and records its tool calls.

Upload a file and wait for “Ready to search.” With no sources selected, chat searches all ready files; select individual library entries to narrow the scope. Preferences affect answer presentation. English is the retrieval baseline; output translation is supported, but multilingual retrieval has not been evaluated.

## MVP limits

PDF/TXT: 10 MB, PDF up to 100 pages. Images: 5 MB and 20 megapixels. Videos: 50 MB, up to two minutes, at most six frames. Supported extensions: PDF, TXT, PNG, JPEG, WebP, MP4, MOV, WebM. Scanned PDF OCR, audio transcription, CLIP image retrieval, scene detection, durable job queues, and deployment are deferred. Brief visual events can be missed. The assistant cannot answer questions about spoken video content because audio is not indexed.

Ingestion runs in the backend with concurrency one. Restarting marks interrupted jobs failed; delete/reupload to retry. Deletion removes local source files and associated indexes. Selected text and images are sent to Groq; originals, embeddings, and metadata otherwise stay in local `data/`.

## Verification

```powershell
# With backend running; makes bounded live Groq calls using synthetic fixtures.
.venv/Scripts/python.exe scripts/smoke_test.py
```

The test creates a PDF, an image, and a short video; checks extraction, both hybrid channels, paraphrase retrieval, source filtering, FTS escaping, all three MCP tools, grounded answers, abstention, source resolution, video range requests, and deletion. It removes its uploaded test sources after success and writes results to ignored `data/test-fixtures/smoke-results.json`. Live tests consume free-tier quota and may need a retry if Groq is throttling.

The UI's TypeScript checks and production compilation run with `npm run build`. Browser visual QA may be limited by the local browser tooling; API smoke tests do not establish pixel-level UI correctness.

To test your own files explicitly, run `scripts/test_files.py --pdf <path> --video <path>` with the virtual-environment Python. This sends retrieved text and frames to Groq, leaves those sources in the local library, and saves private answers in ignored `data/user-acceptance.json`. Run `scripts/check_staged.py` before committing to check staged files for the configured key and private runtime folders.

## Project reference

See [the system design](docs/SYSTEM_DESIGN.md). Its delivery-scope section governs the MVP; later sections describe the larger target architecture. The implementation uses FastEmbed's ONNX MiniLM instead of PyTorch sentence-transformers to reduce dependency size.

`.env`, data, model caches, `.venv`, `node_modules`, and frontend build output are excluded from Git. `.env.example` contains placeholders only. Never put the API key in React variables or commit it. Copy the template only for first-time setup.
