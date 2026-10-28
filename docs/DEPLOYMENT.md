# Deploying DriveDocs AI to Render (Free plan)

DriveDocs AI is one FastAPI process that serves the REST API and the web UI. It deploys as a **single Render Web
Service on the Free plan**, defined by [`render.yaml`](../render.yaml): no database, worker or extra service is needed.
A [`Dockerfile`](../Dockerfile) is provided for other container platforms.

> **Status: deployed** at <https://drivedocs-3xr4.onrender.com>. The v0.3 changes (LangChain agent, vehicle-manual
> corpus, guardrails) deploy automatically when they are pushed to `main`; afterwards run the checks in
> [§5](#5-verify-the-public-website-after-each-deploy).

## 1. What was verified before deployment (v0.3)

| Check | Result | How |
| --- | --- | --- |
| Clean production install | ✅ | Fresh virtualenv, `pip install -r requirements.txt` (adds `langchain-core 1.7.0`), **no PyTorch** |
| Render's platform can install every dependency | ✅ | All pinned packages publish Linux x86-64 wheels for CPython 3.14 (`langchain-core` and its dependencies are pure Python or ship cp314 wheels) |
| Automated tests | ✅ | `pytest`: **207 passed** (mock embeddings, incl. 9 `real_models` tests against the cached ONNX models); the same suite plus the real-model evaluation runs in GitHub Actions CI |
| Build step | ✅ | `HF_HUB_OFFLINE=0 python scripts/download_models.py`: int8 MiniLM (23 MB) + 4-bit CLIP (183 MB) |
| Runtime needs no network | ✅ | Started with `HF_HUB_OFFLINE=1` and the proxy unreachable; models load from the build cache |
| Port binding & health | ✅ | `/health` answers immediately; ready after ~0.5 s locally from the prebuilt index (Render's shared CPU: expect a few seconds) |
| End-to-end API | ✅ | [`scripts/smoke_test.py`](../scripts/smoke_test.py): **30/30 checks** (health, ready, UI, agent and guardrail settings, corpus incl. redacted service record, text search, cited answer, agent tool calls, compound question, refusal, prompt-injection block, PII refusal, no PII in service-record answers, text→image and image→image search, engine-bay diagram via the agent, 4 invalid uploads, path traversal, evaluation, `/docs`) |
| Memory | ✅ | ~318 MB RSS at start-up from the prebuilt index with the agent loaded, ~322 MB after repeated agent chats, ~388 MB while the CLIP vision tower embeds an image, back to ~320 MB after it is released (Windows; v0.2 reported 489 MB on Render) |
| Frontend ↔ backend | ✅ | Headless Edge: agent trace, verified citations, diagram answers, guardrail badges, evaluation and security dashboards, mobile layout |
| Evaluation | ✅ | QA 39/39 (Recall@5 100%, groundedness 100%), security 23/23 (0% attack success, 0% PII leaks); p50 6.6 ms |
| Linux memory for v0.3 | ⚠️ not measured locally | No Linux/Docker environment was available; check `/ready` → `memory_rss_mb` after the deploy |
| `docker build` | ⚠️ not run | Docker is not installed on the development machine |

## 2. Resources: why the Free plan works

Embeddings run on **ONNX Runtime** instead of PyTorch: MiniLM with int8 weights and CLIP ViT-B/32 with 4-bit weights
(the end-to-end evaluation is identical to fp32 MiniLM). Details and the variant comparison are in
[RAG_DESIGN.md §10](RAG_DESIGN.md#10-embedding-runtime-onnx-instead-of-pytorch).

| Configuration | RSS after start-up (local) | Render plan |
| --- | --- | --- |
| **v0.3 default: agent + text + image search, int8 MiniLM, prebuilt index, vision tower idle** | ~320 MB | **Free (512 MB)** |
| v0.3 with fp32 MiniLM (`DRIVEDOCS_TEXT_ONNX_FILE=onnx/model.onnx`) | ~463 MB | Free, little headroom |
| Text only (`DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`) | ~250 MB | Free, large headroom |
| Previous PyTorch stack, for comparison | 864 MB | needed Standard (2 GB, paid) |

The int8 switch exists because LangChain adds ~37 MB and the v0.2 build already reported 489 MB on Render. After
start-up the service releases the CLIP vision tower (only needed to embed new images) and calls `malloc_trim(0)` on Linux so freed memory is returned to the OS. The build runs `scripts/build_demo_index.py`, so instances start from a ready index instead of embedding the samples.

Extra safety settings in `render.yaml` for the small shared instance: one uvicorn worker, at most one inference at a
time (`DRIVEDOCS_MAX_CONCURRENT_INFERENCE=1`), two ONNX threads, `MALLOC_ARENA_MAX=2`, 10 MB uploads and at most 100
documents.

**If the free instance runs out of memory** (Render logs "Ran out of memory" and restarts it): set
`DRIVEDOCS_ENABLE_IMAGE_SEARCH=false` in the dashboard (~250 MB), or change `plan:` to `starter`/`standard`.

Free-plan behaviour to know about: the service **sleeps after 15 minutes without traffic** and the next visitor
waits ~1 minute while it wakes; Render shows a loading page meanwhile. Free services get 750 instance hours per
month.

## 3. Storage: what persists?

Render's filesystem is **ephemeral**: local changes are lost on every restart, redeploy or sleep.

| Data | Free plan (no disk) | Paid plan with a persistent disk |
| --- | --- | --- |
| Bundled sample documents (`data/sample/`) | Re-indexed automatically on every boot | Indexed once, kept |
| Documents uploaded by visitors, their images and index entries | **Lost** on restart/redeploy/sleep | Kept |
| Model files (`HF_HOME` in the project directory) | Downloaded at each **build**, kept for that deploy | Same |
| Evaluation report (`reports/`) | Shipped with the repository | Same |

The public demo is designed around the bundled samples. A permanent banner in the UI states that the demo has no
authentication and that uploads are deleted on restart (`/system/info` → `storage_mode: "ephemeral"`).

**To keep uploads:** switch to a paid plan, uncomment the `disk:` block in `render.yaml` (mount `/var/data`), set
`DRIVEDOCS_STORAGE_DIR=/var/data/drivedocs` and `DRIVEDOCS_PERSISTENT_STORAGE=true`, and redeploy.

## 4. Deploy step by step

**Prerequisite: push the code.** Render builds from GitHub: commit and push the changes (`git add -A`, `git commit`,
`git push origin main`). The existing service redeploys automatically; the steps below create a new one.

1. Open <https://dashboard.render.com/> and sign in with GitHub. Never paste API keys or passwords into chats or files.
2. Click **New → Blueprint**.
3. Select the repository **Charanloyal/drivedocs** (click "Configure account" to grant Render access if it is not
   listed).
4. Branch: **main**. Blueprint path: `render.yaml` (default). Give the Blueprint any name, e.g. `drivedocs`.
5. Render lists one resource: **web service `drivedocs` (Free)**. You are prompted for `GEMINI_API_KEY` and
   `OPENAI_API_KEY`: **leave both empty** (the keyless extractive answerer is the default).
6. Click **Deploy Blueprint**.
7. Open the service and watch **Logs**. A healthy build shows (Render's wording may differ slightly):
   * Python **3.14.7** being used
   * `Successfully installed … onnxruntime-1.31.0 … faiss-cpu-1.15.1 …`
   * `[*] Embedding backend: onnx; image search: True`
   * `[+] Xenova/all-MiniLM-L6-v2: 3 files, 23 MB …` and `[+] Xenova/clip-vit-base-patch32: 5 files, 183 MB …`
   * `[+] Demo index: 5 documents, 40 chunks, 5 images in /opt/render/project/src/data/prebuilt (…s)`
   * a "Build successful" message
8. Then the runtime log should show (no "Ingested" lines: the index comes from the build):
   ```text
   Starting DriveDocs 0.3.0 (device=cpu, storage=ephemeral, image_search=True, provider=extractive)
   Uvicorn running on http://0.0.0.0:10000
   Restored the prebuilt demo index from prebuilt
   Loaded model Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx|onnx|text in …s
   Loaded model Xenova/clip-vit-base-patch32/onnx/text_model_q4.onnx+onnx/vision_model_q4.onnx|onnx|clip in …s
   DriveDocs ready in … ms: 5 documents, 40 chunks, 5 images, storage=ephemeral, prebuilt index=True
   ```
   and the status becomes **Live**. The URL is at the top of the service page: `https://drivedocs.onrender.com` if
   the name is free, otherwise Render adds a suffix such as `https://drivedocs-ab12.onrender.com`.

**Without a Blueprint** (New → Web Service): runtime *Python 3*, branch `main`, plan *Free*, and copy the
`buildCommand`, `startCommand`, health check path `/health` and every environment variable from `render.yaml`.

### Troubleshooting

| Symptom in the logs | Cause / fix |
| --- | --- |
| A push to `main` does not change the live site | **Settings → Build & Deploy → Auto-Deploy** is off (turn it on, or use **Manual Deploy → Deploy latest commit**), or the build failed (see **Events** for a red "Deploy failed" and open its log) |
| `/ready` → `timings_ms` contains `seed_samples` (start-up indexing) instead of `restore_prebuilt_index` | The dashboard's **Build Command** lacks `&& python scripts/build_demo_index.py`. Services created in the dashboard (not from the Blueprint) do not pick up `buildCommand` changes from `render.yaml`; copy it from there |
| `[!] Demo index build failed` / `Expected 5 ready documents` | A sample file is missing or unreadable in the build; the log line names it |
| The build installs `torch` or `sentence-transformers` | Render is building the old code: push the new commit and use **Manual Deploy → Deploy latest commit** |
| `Ran out of memory (used over 512MB)` | Set `DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`, or use a paid plan (§2) |
| `OfflineModeIsEnabled` during the build | The build command lost its `HF_HUB_OFFLINE=0` prefix; restore it from `render.yaml` |
| `429` / timeouts from huggingface.co during the build | Hub rate limit: retry, or add a read-only `HF_TOKEN` secret in the dashboard |
| `Model '…' could not be loaded` and the health check fails | The build-time model cache is missing; check `HF_HOME` is unchanged and the download step ran |
| `No open ports detected` | The start command was edited; it must contain `--host 0.0.0.0 --port $PORT` |
| Page takes ~1 minute to load | Normal on the Free plan after 15 idle minutes (the service is waking up) |
| `No matching distribution found for …` | Keep `PYTHON_VERSION=3.14.7` (the pinned wheels were verified for it) |

## 5. Verify the public website after each deploy

From the repository folder (standard library only; waits up to 4 minutes for a sleeping service to wake):

```bash
python scripts/smoke_test.py https://drivedocs-3xr4.onrender.com
```

All checks must pass (`30/30 checks passed`). Then check the UI by hand:

1. Open <https://drivedocs-3xr4.onrender.com/>. The status pill turns green ("Ready · 40 chunks"); the status card
   lists "Agent (LangChain): on" and "Guardrails: 3/3 on".
2. Click "What is the recommended cold tire pressure…": the answer contains **35 psi** with a clickable
   `aurora_s5_owner_manual.pdf · p.2` citation; the inspector shows the agent's tool calls.
3. Click the compound brake-fluid example: two sub-questions in the trace, "3 years" and "DOT 4" in the answer.
4. Click "Where is the coolant reservoir?": the engine-bay diagram is attached.
5. Click the two guardrail examples: both are "Blocked by guardrails".
6. **Library** → `service_record_ro58213.txt`: personal fields show `[REDACTED_…]`.
7. **Evaluation**: Recall@5 and groundedness 100%, security table all "pass".
8. `/ready` → `memory_rss_mb` stays well below 512.

## 6. Environment variables

All are set by `render.yaml`. **Nothing has to be entered manually** for the demo. Optional values:

| Variable | Where | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | Dashboard (secret, `sync: false`) | Only with `LLM_PROVIDER=gemini` / `openai`; leave empty otherwise. An LLM agent planner additionally needs `langchain-openai` / `langchain-google-genai` added to the build |
| `HF_TOKEN` | Dashboard (secret), optional | Only if model downloads during the build hit Hub rate limits |
| `DRIVEDOCS_CORS_ORIGINS` | Dashboard, optional | Only if another website must call the API from the browser |

Defined in `render.yaml`: `PYTHON_VERSION=3.14.7`, `DEVICE=cpu`, `LLM_PROVIDER=extractive`,
`HF_HOME=/opt/render/project/src/.hf-cache`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`DRIVEDOCS_EMBEDDING_BACKEND=onnx`, `DRIVEDOCS_ENABLE_IMAGE_SEARCH=true`, `DRIVEDOCS_AGENT_ENABLED=true`,
`DRIVEDOCS_REDACT_PII=true`, `DRIVEDOCS_DETECT_PROMPT_INJECTION=true`, `DRIVEDOCS_SEED_SAMPLE_DATA=true`,
`DRIVEDOCS_STORAGE_DIR=/opt/render/project/src/data/storage`, `DRIVEDOCS_PERSISTENT_STORAGE=false`,
`DRIVEDOCS_MAX_UPLOAD_MB=10`, `DRIVEDOCS_MAX_DOCUMENTS=100`, `DRIVEDOCS_ONNX_THREADS=2`,
`DRIVEDOCS_MAX_CONCURRENT_INFERENCE=1`, `OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`, `TOKENIZERS_PARALLELISM=false`.
`PORT` is provided by Render (default 10000). The full list with local defaults is in
[`.env.example`](../.env.example).

## 7. Health checks

| Endpoint | Purpose | Loads models? |
| --- | --- | --- |
| `GET /health` | Liveness for Render: 200 while starting or ready, **503 if start-up failed** | Never |
| `GET /ready` | Readiness: storage, registry, indexes, text model; 503 until ready; includes RSS and start-up timings | Never (reports status) |

## 8. Docker (other platforms)

```bash
docker build -t drivedocs .                                    # ONNX models baked into the image
docker run -p 8000:8000 -v drivedocs-data:/data drivedocs      # add -e DRIVEDOCS_PERSISTENT_STORAGE=true with a volume
```

The image runs as a non-root user, respects `$PORT` and keeps data in `/data/drivedocs`. `docker build` was not
executed during development.

## 9. Security notes for a public demo

* **No authentication.** Every visitor can list, read, search and delete every document. Do not upload private
  documents to a public instance; the UI banner says so.
* Uploads are untrusted: sanitized names, hash-derived storage paths, content-signature checks,
  size/page/pixel/count limits, a streaming body limit, and images served only by registered ID.
* Strict CSP (no inline scripts), `nosniff`, `X-Frame-Options: DENY`, `no-referrer`; document text is never
  rendered as HTML. Errors never include stack traces or paths. Logs never include document contents or keys.
* Secrets are only read from environment variables; `render.yaml` declares them with `sync: false` and contains no
  values.
* **Guardrails:** personal data is redacted at ingestion and in every response, questions asking for personal data
  are refused, and instruction-like text in questions or documents is removed, never quoted and never sent to an LLM
  (see the README's *Security guardrails* section and the security suite in `reports/EVALUATION.md`).
