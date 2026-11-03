# DriveDocs AI: Audit (v0.3, November 2026)

Audited revision: `fc34b2b` (`main`, deployed at <https://drivedocs-3xr4.onrender.com>).
Method: read the code paths listed below, ran the test suite (208 passed), a one-off lint (`ruff --select F,E9,B`) and
dependency scan (`pip-audit -r requirements.txt`), started the app locally with the real models, exercised the API and
UI (headless Edge at 360/390/768/1440 px), and queried the live deployment's `/ready` and `/system/info`.

Each finding is marked **verified** (reproduced or measured) or **suspected** (reasoned from the code, not
reproduced). The fix column says what this revision does about it.

## Summary

No critical defect was found: the core journeys (upload, retrieval, agent answers, citations, guardrails, deletion,
persistence) work and are covered by tests, CI and the deployed smoke test. The important problems are a vulnerable
PDF parser, two abuse paths on the public demo, a deploy-configuration drift, and in-sample evaluation.

| # | Severity | Area | Finding | Evidence | Fix in this revision |
| --- | --- | --- | --- | --- | --- |
| 1 | High | Security / dependencies | `pypdf 6.21.2` has 7 published CVEs (crafted PDFs cause infinite loops or runaway memory; one is in the page-count path every upload uses). Uploaded PDFs are untrusted, and the demo runs on a 512 MB instance | **verified**: `pip-audit` | Upgraded to `pypdf 6.21.4` (fixes all 7) |
| 2 | High | Security / availability | Any visitor can delete the bundled sample documents (`DELETE /documents/{id}`), breaking the demo for everyone until the next restart | **verified**: deleting `aurora_s5_owner_manual.pdf` locally returned 200 | Bundled samples are read-only (403 `sample_document_protected`) unless `DRIVEDOCS_ALLOW_SAMPLE_DELETION=true`; UI hides the delete buttons for samples |
| 3 | Medium | Security / abuse | No rate limiting: 40 back-to-back `/chat` requests all succeeded; uploads (up to 10 MB / 300 pages each) are unthrottled on a single small instance | **verified** | Per-client sliding-window limits on chat, search, upload and delete; `429 rate_limited` with `Retry-After` |
| 4 | Medium | Deployment | The live build never produces the prebuilt demo index, so every start-up re-embeds the samples (~21 s of a ~34 s start-up on Render). Cause: the Render Blueprint is not syncing, so the dashboard build command lacks `build_demo_index.py` | **verified**: live `/ready` → `prebuilt_index: not found: data/prebuilt/ does not exist` | `scripts/download_models.py` (already in the live build command) now also builds the demo index, so no dashboard change is needed |
| 5 | Medium | Evaluation | All reported QA results were in-sample: thresholds and fixes were tuned on the same 39 questions, and there was no held-out set | **verified** (documented in RAG_DESIGN §9) | Added a 23-question held-out set written after tuning, run once without changes afterwards: **78.3% pass (18/23)** vs. 100% in-sample; reported separately |
| 5a | Medium | RAG quality | Near-domain questions about something the documents do not contain get a confident answer built from related evidence (3 of 4 held-out cases: "towing capacity" → how to tow; "TPMS reset" → what the warning means; "hybrid oil" → the regular oil spec). Citations are valid, but the answer implies it addresses the question. The "not mentioned" flag only covers names, acronyms and model numbers | **verified** (held-out set) | **Not fixed on purpose:** a fix designed against these held-out questions would invalidate them as a held-out measure. Next step: a question-coverage check, validated on a new held-out set |
| 5b | Low | RAG quality | Short paraphrases can fall under the 0.30 text gate even when the right passage ranks first ("How big is the fuel tank?" → top score 0.236 → refused) | **verified** (held-out set) | Not fixed on purpose (same reason); documented |
| 6 | Medium | UI / responsive | The Library page scrolls horizontally on phones and tablets (page width 666 px on a 360 px screen, 819 px at 768 px): the table card cannot shrink inside the flex column | **verified** (headless Edge) | Flex children may shrink (`min-width: 0`), so the table scrolls inside its card |
| 7 | Medium | Robustness | PDF parsing runs while holding the service-wide write lock with no time budget. An as-yet-unknown parser hang would block all uploads and deletions (chat keeps working: index reads use their own locks) | **suspected**, not reproduced | Not changed (process isolation would cost memory on the Free plan); mitigated by #1 and the page/size limits; documented |
| 8 | Low | UI | On phones the top navigation hides the "Evaluation" tab off-screen (the tab bar scrolls, with no visual cue) | **verified** | Tabs wrap onto two lines on narrow screens |
| 9 | Low | Observability | Errors and log lines cannot be correlated: no request ID | **verified** | `X-Request-ID` on every response (accepted from the caller if well-formed), included in error bodies and access logs |
| 10 | Low | Operations | Render Blueprint automatic sync is off, so `render.yaml` changes (build command, env vars, `autoDeployTrigger`) are not applied | **verified** (dashboard build command, deploy history) | Documented in DEPLOYMENT.md; #4 removes the dependency for the index |
| 11 | Low | Code hygiene | Unused imports in 4 test modules; `zip()` without `strict=` | **verified**: ruff | Unused imports removed |
| — | Limitation | Security model | No authentication or per-user isolation: visitors can see and delete each other's uploads; names written in free prose (outside labelled fields) are not detected as PII; pattern-based injection screening cannot catch every paraphrase | by design for a public demo; documented | Unchanged; documented in the README security section |

## Checked and found sound

* **Concurrency:** FAISS text and image indexes guard add/remove/search with locks; ingestion and deletion run under
  one write lock and roll back on failure (tests cover rollback and crash recovery).
* **Uploads:** names sanitized, storage paths derived from content hashes, magic-byte checks, size/page/pixel/image
  limits, streaming body limit; path traversal on the image route returns 404 (smoke-tested live).
* **XSS / headers:** the UI renders all document text with `textContent` under a strict CSP; `nosniff`,
  `X-Frame-Options: DENY`, `no-referrer`; CORS off by default.
* **Errors:** unhandled exceptions return a generic `internal_error`; no stack traces or paths in responses; `/ready`
  reports project-relative paths only.
* **PII:** redacted before embedding; re-scrubbed in answers, passages, section headings and citation snippets;
  logs contain document IDs, not content.
* **Prompt injection:** retrieved text is treated as untrusted (never quoted when instruction-like, stripped from LLM
  prompts, delimited); the planner's prose is never used as the answer.
* **LLM honesty:** `provider` reports `extractive` whenever an LLM is not configured or its answer was replaced.
* **Secrets:** none in the repository (pattern scan for OpenAI/Google/GitHub/Render keys and private keys);
  `render.yaml` declares API keys with `sync: false`.
* **Lint:** no undefined names, syntax errors or real bug patterns (`ruff` B008 hits are FastAPI's dependency idiom).
