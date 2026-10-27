# Deploying DriveDocs AI to Render (Free plan)

DriveDocs AI is one FastAPI process that serves the REST API and the web UI. It runs as a **single Render Web Service
on the Free plan**, defined by [`render.yaml`](../render.yaml), with no database, worker or extra service. A
[`Dockerfile`](../Dockerfile) is provided for other container platforms.

**Live instance:** <https://drivedocs-3xr4.onrender.com> (API docs: <https://drivedocs-3xr4.onrender.com/docs>).
It deploys automatically from `main`. Verification results are in [§5](#5-verify-the-public-website).

## 1. What was verified locally

| Check | Result | How |
| --- | --- | --- |
| Clean production install | ✅ | Fresh virtualenv, `pip install -r requirements.txt` + `pip install --no-deps -e .`; `pip check` clean; **no PyTorch** installed |
| Linux wheels for every pin | ✅ | `pip download --platform manylinux2014_x86_64 --python-version 3.14` resolves all 68 wheels (~102 MB), including `onnxruntime`, `faiss-cpu`, `onnx` and `langchain-google-genai` |
| Automated tests | ✅ | `pytest`: **170 passed** (mock embeddings + scripted fake chat model for the agent) |
| Build steps | ✅ | `download_models.py` (ONNX files + int8 variants) then `build_demo_index.py`: 5 documents, 26 chunks, 5 images in 1.8 s |
| Start command | ✅ | Exact `startCommand` with the `render.yaml` variables and `PORT=10000`: `/health` 200 in 18 ms, **ready 1.2 s** after start (prebuilt index, no re-indexing) |
| Start-up failure is detected | ✅ | With the model cache missing, the log names the cause and `/health` returns **503**, so Render fails the deploy instead of serving a broken app |
| End-to-end API | ✅ | [`scripts/smoke_test.py`](../scripts/smoke_test.py): **24/24 checks** (see §5) |
| Frontend ↔ backend | ✅ | Headless Chrome walkthrough: cited answers, refusal, PII badge, diagram search with captions, upload of a hostile file name, XSS payloads rendered as text, evaluation dashboard, mobile layout; no console errors |
| Evaluation | ✅ | 26/26 QA queries pass, Recall@5 96.4%, groundedness 100%; security suite 10/10 ([report](../reports/EVALUATION.md)) |
| Agent with a real Gemini key | ⏳ | No key on the development machine. The agent is covered by unit tests with a scripted LangChain chat model; it runs live once `GEMINI_API_KEY` is set (§4) |
| `docker build` | ⚠️ not run | Docker is not installed on the development machine |

## 2. Resources: why the Free plan works

The Free instance has **512 MB of RAM** and a shared CPU. DriveDocs fits by avoiding PyTorch entirely:

* **ONNX Runtime** runs MiniLM (fp32 weights) and CLIP ViT-B/32 (4-bit weights). Vectors match the PyTorch models
  (MiniLM cosine 1.0000) and the evaluation is unchanged; see
  [RAG_DESIGN.md §9](RAG_DESIGN.md#9-embedding-runtime-onnx-instead-of-pytorch).
* **int8 embedding tables.** `download_models.py` quantizes the large token-embedding tables of both graphs
  ([`onnx_optimize.py`](../src/drivedocs/onnx_optimize.py)). The MiniLM file shrinks from 86 to 53 MB and its RAM
  from 116 to 64 MB; the CLIP text file shrinks from 120 to 48 MB. Output cosine to the original is ≥ 0.9998.
* **Lazy CLIP vision tower.** Text→image search only needs the CLIP *text* tower. The vision tower (~70 MB) loads
  only for image uploads and image→image search, and is unloaded after `DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS`
  (120 s on Render).
* **Build-time demo index.** The sample corpus is indexed during the build, so boot does no ingestion work.
* **LangChain is imported lazily.** `langchain-core` and `langchain-google-genai` (~40 MB) load only when a Gemini
  key is configured.

Measured private memory per component (development machine, `psutil` private bytes):

| Component | RAM |
| --- | --- |
| Python + FastAPI + FAISS + ONNX Runtime + app | ~117 MB |
| MiniLM (int8 embedding table) | ~64 MB |
| CLIP text tower (q4) | ~127 MB before int8 tables, less after |
| CLIP vision tower (q4), only while in use | ~69 MB |
| LangChain + Gemini client, only with a key | ~41 MB |

<!-- LIVE_MEMORY -->

Other safety settings in `render.yaml`: one uvicorn worker, at most one inference at a time, two ONNX threads,
`MALLOC_ARENA_MAX=2`, 10 MB uploads and at most 100 documents.

**If the instance runs out of memory** (Render logs "Ran out of memory" and restarts it): set
`DRIVEDOCS_ENABLE_IMAGE_SEARCH=false` in the dashboard (saves ~0.2 GB), or change `plan:` to `starter`.

Free-plan behaviour: the service **sleeps after 15 minutes without traffic**, and the next visitor waits about a
minute while it wakes. Free services get 750 instance hours per month.

## 3. Storage: what persists?

Render's filesystem is **ephemeral**: runtime changes are lost on every restart, redeploy or sleep.

| Data | Free plan (no disk) | Paid plan with a persistent disk |
| --- | --- | --- |
| Bundled sample documents (`data/sample/`) | Indexed during the build; back after every restart | Indexed once, kept |
| Documents uploaded by visitors, their images and index entries | **Lost** on restart/redeploy/sleep | Kept |
| Model files (`HF_HOME` in the project directory) | Downloaded at each **build**, kept for that deploy | Same |
| Evaluation report (`reports/`) | Shipped with the repository | Same |

The UI shows a permanent banner saying that the demo has no authentication and that uploads are deleted on restart
(`/system/info` → `storage_mode: "ephemeral"`). Uploaded files are never promised to persist.

**To keep uploads:** switch to a paid plan, uncomment the `disk:` block in `render.yaml` (mount `/var/data`), set
`DRIVEDOCS_STORAGE_DIR=/var/data/drivedocs` and `DRIVEDOCS_PERSISTENT_STORAGE=true`, and redeploy.

## 4. Deploy / update step by step

### Updating the existing service

`main` auto-deploys. After `git push origin main`, open the service in <https://dashboard.render.com/> and watch
**Events → Deploy** and **Logs**.

**Enable the LangChain agent (one-time, manual).** Never paste keys into chats, issues or files.

1. Create a free API key in [Google AI Studio](https://aistudio.google.com/apikey).
2. In the Render dashboard, open the **drivedocs** service → **Environment** → **Add Environment Variable**.
   Set key `GEMINI_API_KEY` and paste the key as the value.
3. Check that `LLM_PROVIDER` is `agent`. A Blueprint sync sets it; a service created manually may still have the
   old `extractive` value.
4. Click **Save, rebuild, and deploy**.
5. Verify: `/system/info` shows `"llm_configured": true` and `"llm_provider": "LangChain agent (gemini-3.5-flash-lite)"`.
   An answer in the UI shows "Agent steps: 1. 🔎 search_manual_text(…)".

Without the key, everything works with the keyless extractive answerer. The same happens if Gemini rate-limits
or fails: the answer falls back to extractive and carries a warning.

**If the service was created as a plain Web Service** (not a Blueprint), `render.yaml` changes are not applied
automatically. Under **Settings**, set the build command to:

```text
pip install --upgrade pip && pip install -r requirements.txt && pip install --no-deps -e . && HF_HUB_OFFLINE=0 python scripts/download_models.py && python scripts/build_demo_index.py
```

Then copy any new variables from `render.yaml` under **Environment**: `LLM_PROVIDER=agent`,
`GEMINI_MODEL=gemini-3.5-flash-lite` and `DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS=120`. The old build command still
works, but every boot then re-indexes the samples, which costs ~10 s and some memory.

### Creating a new service

1. Open <https://dashboard.render.com/>, then click **New → Blueprint**.
2. Select **Charanloyal/drivedocs** (use "Configure account" to grant access), branch **main**, Blueprint path
   `render.yaml`.
3. Render lists one resource: **web service `drivedocs` (Free)**. When prompted for `GEMINI_API_KEY`, paste your
   key, or leave it empty for extractive-only answers.
4. Click **Deploy Blueprint** and watch **Logs**. A healthy build shows Python 3.14.7, `Successfully installed …
   onnxruntime … faiss-cpu … langchain-google-genai …`, the model downloads followed by `[+] optimized … ->
   …emb8.onnx` lines, then `[+] Demo index: 5 documents, 26 chunks, 5 images`. The runtime log shows:
   ```text
   Starting DriveDocs 0.3.0 (device=cpu, storage=ephemeral, image_search=True, provider=LangChain agent (gemini-3.5-flash-lite))
   Uvicorn running on http://0.0.0.0:10000
   DriveDocs ready in … ms: 5 documents, 26 chunks, 5 images, storage=ephemeral
   ```
   The status then becomes **Live**. The URL is shown at the top of the service page.

### Troubleshooting

| Symptom in the logs | Cause / fix |
| --- | --- |
| `Ran out of memory (used over 512MB)` | Set `DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`, or use a paid plan (§2) |
| `OfflineModeIsEnabled` during the build | The build command lost its `HF_HUB_OFFLINE=0` prefix; restore it from `render.yaml` |
| `429` / timeouts from huggingface.co during the build | Hub rate limit: retry, or add a read-only `HF_TOKEN` secret in the dashboard |
| `Model '…' could not be loaded` and the health check fails | The build-time model cache is missing; check that `HF_HOME` is unchanged and the download step ran |
| Answers say "The LLM agent was unavailable (…)" | The Gemini key is invalid, the model name is wrong, or the quota is exhausted; the extractive answer is shown meanwhile |
| `No open ports detected` | The start command was edited; it must contain `--host 0.0.0.0 --port $PORT` |
| Page takes about a minute to load | Normal on the Free plan after 15 idle minutes |
| `No matching distribution found for …` | Keep `PYTHON_VERSION=3.14.7` (the pinned wheels were verified for it) |

## 5. Verify the public website

From the repository folder (standard library only; waits up to 4 minutes for a sleeping service to wake):

```bash
python scripts/smoke_test.py https://drivedocs-3xr4.onrender.com
python scripts/run_eval.py --base-url https://drivedocs-3xr4.onrender.com --output-dir reports/live
```

The smoke test runs 24 checks:
- health, readiness, the UI and its CSP;
- the document list;
- a text search that finds manual page 5;
- a cited tire-pressure answer (2.7 bar), with every citation on a real file and page;
- the answer provider;
- PII masking of the service record;
- refusal of an off-topic question;
- text→image search returning the fuse-box diagram;
- image→image search;
- a diagram answer;
- 4 invalid uploads, path traversal, `/evaluation` and `/docs`.

The remote evaluation uploads the security documents, runs the QA set and security suite over HTTP, and then
deletes them.

<!-- LIVE_VERIFICATION -->

Manual checks in the browser:

1. Open the site. The status pill turns green ("Ready · 26 chunks").
2. Click "Manual · page 5 + fuse diagram". The answer says **F12 (15 A)** with a verified
   `aurora_owners_manual.pdf · p.5` citation.
3. Click "PII redaction". The answer shows `[REDACTED EMAIL]` and `[REDACTED PHONE]` and a "PII values masked" badge.
4. Click "Not in the manual". The response is "Insufficient evidence" with no citations.
5. Open **Visual search** and search "fuse box layout diagram". The page 5 figure comes first ("Its caption matches
   your request").

## 6. Environment variables

| Variable | Where | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | **Dashboard only** (secret, `sync: false`) | Enables the LangChain agent; leave unset for extractive-only answers |
| `HF_TOKEN` | Dashboard (secret), optional | Only if model downloads during the build hit Hub rate limits |
| `DRIVEDOCS_CORS_ORIGINS` | Dashboard, optional | Only if another website must call the API from the browser |

Defined in `render.yaml`:
- `PYTHON_VERSION=3.14.7`, `DEVICE=cpu`;
- `LLM_PROVIDER=agent`, `GEMINI_MODEL=gemini-3.5-flash-lite`;
- `HF_HOME=/opt/render/project/src/.hf-cache`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`;
- `DRIVEDOCS_EMBEDDING_BACKEND=onnx`, `DRIVEDOCS_ENABLE_IMAGE_SEARCH=true`, `DRIVEDOCS_SEED_SAMPLE_DATA=true`;
- `DRIVEDOCS_STORAGE_DIR=/opt/render/project/src/data/storage`, `DRIVEDOCS_PERSISTENT_STORAGE=false`;
- `DRIVEDOCS_MAX_UPLOAD_MB=10`, `DRIVEDOCS_MAX_DOCUMENTS=100`;
- `DRIVEDOCS_VISION_IDLE_UNLOAD_SECONDS=120`, `DRIVEDOCS_ONNX_THREADS=2`, `DRIVEDOCS_MAX_CONCURRENT_INFERENCE=1`;
- `OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`, `TOKENIZERS_PARALLELISM=false`.

`PORT` is provided by Render (default 10000). The full list with local defaults is in
[`.env.example`](../.env.example).

## 7. Health checks

| Endpoint | Purpose | Loads models? |
| --- | --- | --- |
| `GET /health` | Liveness for Render: 200 while starting or ready, **503 if start-up failed** | Never |
| `GET /ready` | Readiness of storage, registry, indexes and models; 503 until ready; includes RSS (`memory_rss_mb`) and start-up timings | Never (reports status) |

## 8. Docker (other platforms)

```bash
docker build -t drivedocs .                         # ONNX models and the demo index baked into the image
docker run -p 8000:8000 -e GEMINI_API_KEY=... -v drivedocs-data:/data drivedocs
```

The image runs as a non-root user, respects `$PORT` and keeps data in `/data/drivedocs` (add
`-e DRIVEDOCS_PERSISTENT_STORAGE=true` when a volume is mounted). `docker build` was not executed during development.

## 9. Security notes for a public demo

* **No authentication.** Every visitor can list, read, search and delete every document. Do not upload private
  documents to a public instance; the UI banner says so.
* **PII masking and prompt-injection defenses** apply to every response (see the README). They reduce risk but are
  not a substitute for access control.
* Uploads are untrusted: sanitized names, hash-derived storage paths, content-signature checks,
  size/page/pixel/count limits, a streaming body limit, and images served only by registered ID.
* Strict CSP (no inline scripts), `nosniff`, `X-Frame-Options: DENY`, `no-referrer`. Document text is never rendered
  as HTML, errors never include stack traces or paths, and logs never include document contents or keys.
* Secrets are read only from environment variables. `render.yaml` declares `GEMINI_API_KEY` with `sync: false` and
  contains no values.
