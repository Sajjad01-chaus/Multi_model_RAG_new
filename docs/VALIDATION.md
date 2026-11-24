# MVP validation and known limits

Verified locally on 24 November 2026.

- React TypeScript checks and Vite production build passed.
- Backend Python compilation and built frontend HTTP response passed.
- Initial live suite passed 33 checks across PDF/image/video ingestion, dense and BM25 retrieval, rank fusion, exact and paraphrased queries, source filtering, FTS escaping, all three real MCP tools, grounded responses, abstention, valid source links, HTTP video range requests, and deletion.
- A subsequent run verified numbered PDF/image citations after presentation fixes, then reached Groq HTTP 429 during video generation. This is a free-tier quota constraint; the backend reports temporary unavailability rather than inventing an answer. The full revised suite has not been recorded as passing.
- Explicitly supplied user PDF: extraction, grounded-answer generation, scoped citations, and source resolution passed. Supplied video: frame extraction and indexing passed; the grounded-answer request was temporarily unavailable under Groq HTTP 429. Their contents and detailed results remain in ignored local `data/`, never in this repository.
- Secret audit rejects the configured API key, common token formats, private runtime directories, and oversized artifacts in staged files.

## Time-constrained limitations

Video speech/audio transcription is not implemented in this submission. Video search and answers use at most six visual samples with capture timestamps. Brief events may be missed, and the assistant cannot establish what was spoken. This is explicitly disclosed in API responses, source previews, README, and the system design.

Scanned-PDF OCR, CLIP retrieval, scene detection, durable job queues, authentication, and deployment are deferred. English is the retrieval baseline. Model output can still contain factual mistakes despite grounding and citation validation; valid IDs alone do not prove claim support. Actual quotas and model access depend on the configured Groq account.

Browser visual/interaction QA was blocked: the in-app browser timed out reaching localhost, and the desktop browser tool stopped because it could not confidently determine the current URL. Successful build and HTTP checks do not replace visual QA.
