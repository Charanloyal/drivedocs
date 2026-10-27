# Deploying DriveDocs AI to Render (Free plan)

DriveDocs AI is one FastAPI process that serves the REST API and the web UI. It deploys as a **single Render Web
Service on the Free plan**, defined by [`render.yaml`](../render.yaml): no database, worker or extra service is needed.
A [`Dockerfile`](../Dockerfile) is provided for other container platforms.

> **Status: not deployed yet.** Everything below was verified locally against the exact Render configuration. A live
> URL exists only after you complete [§4](#4-deploy-step-by-step) and the checks in
> [§5](#5-verify-the-public-website-before-publishing-the-link) pass. Until then the README keeps a clearly marked
> placeholder.

## 1. What was verified before deployment

| Check | Result | How |
| --- | --- | --- |
| Clean production install | ✅ | Fresh virtualenv, `pip install -r requirements.txt`: 37 s, 270 MB of packages, **no PyTorch**; `pip check` clean |
| Render's platform can install every dependency | ✅ | All pinned packages, including `onnxruntime 1.31.0` and `faiss-cpu 1.15.1`, publish Linux x86-64 wheels for CPython 3.14 (checked on PyPI) |
| Automated tests | ✅ | `pytest`: **121 passed** (mock embeddings), **121 passed** (real ONNX models), 119 passed + 2 skipped (PyTorch backend) |
| Build step | ✅ | `HF_HUB_OFFLINE=0 python scripts/download_models.py` into an empty `HF_HOME`: 8 files, 270 MB, ~25 s |
| Runtime needs no network | ✅ | Started with the exact `startCommand`, every `render.yaml` variable, `PORT=10000`, and outbound traffic forced through a dead proxy |
| Port binding & health | ✅ | Listens on `0.0.0.0:$PORT`; `/health` answers immediately; ready after ~1–2.5 s |
| Start-up failure is detected | ✅ | With the model cache missing, the log names the cause and `/health` returns **503**, so Render fails the deploy instead of serving a broken app |
| End-to-end API | ✅ | [`scripts/smoke_test.py`](../scripts/smoke_test.py): **22/22 checks** (health, ready, UI, documents, text search, text→image and image→image search, cited answer with real file/page, refusal, 4 invalid uploads, path traversal, evaluation, `/docs`) |
| Memory under load | ✅ | 428 MB after start-up, **436 MB** after 30 concurrent chats + 9 uploads (60 PDF pages, 6 images) + 5 image searches |
| Frontend ↔ backend | ✅ | Headless Chrome: cited answer (3/3 citations verified), refusal, visual-only answer, upload, XSS payloads rendered as text, no console errors |
| Evaluation | ✅ | 19/19 queries pass with the ONNX models; p50 latency 9.9 ms |
| Linux memory | ⚠️ not measured | No Linux environment was available; figures are from Windows (Linux RSS is usually similar or lower) |
| `docker build` | ⚠️ not run | Docker is not installed on the development machine |
| An actual Render build | ⏳ pending | Requires your Render account (§4) |

## 2. Resources: why the Free plan works

Embeddings run on **ONNX Runtime** instead of PyTorch: the same MiniLM model in fp32 (identical vectors) and CLIP
ViT-B/32 with 4-bit weights (the end-to-end evaluation is unchanged). Details and the variant comparison are in
[RAG_DESIGN.md §9](RAG_DESIGN.md#9-embedding-runtime-onnx-instead-of-pytorch).

| Configuration | RSS after start-up | RSS after stress run | Render plan |
| --- | --- | --- | --- |
| **Text + image search (demo default)** | 428 MB | 436 MB | **Free (512 MB)**, ~75 MB headroom |
| Text only (`DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`) | ~250 MB | n/a | Free, large headroom |
| Previous PyTorch stack, for comparison | 864 MB | ~1,000 MB | needed Standard (2 GB, paid) |

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

**Prerequisite: push the code.** Render builds from GitHub, and the old version on `main` is what fails. Commit and
push the reviewed changes first (`git add -A`, `git commit`, `git push origin main`).

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
   * `[+] Xenova/all-MiniLM-L6-v2: 3 files, 87 MB …` and `[+] Xenova/clip-vit-base-patch32: 5 files, 183 MB …`
   * a "Build successful" message
8. Then the runtime log should show:
   ```text
   Starting DriveDocs 0.2.0 (device=cpu, storage=ephemeral, image_search=True, provider=extractive)
   Uvicorn running on http://0.0.0.0:10000
   Loaded model Xenova/all-MiniLM-L6-v2/onnx/model.onnx|onnx|text in …s
   Loaded model Xenova/clip-vit-base-patch32/onnx/text_model_q4.onnx+onnx/vision_model_q4.onnx|onnx|clip in …s
   Ingested doc_… (pdf): 6 chunks, 1 images      (×4 sample files)
   DriveDocs ready in … ms: 4 documents, 11 chunks, 2 images, storage=ephemeral
   ```
   and the status becomes **Live**. The URL is at the top of the service page: `https://drivedocs.onrender.com` if
   the name is free, otherwise Render adds a suffix such as `https://drivedocs-ab12.onrender.com`.

**Without a Blueprint** (New → Web Service): runtime *Python 3*, branch `main`, plan *Free*, and copy the
`buildCommand`, `startCommand`, health check path `/health` and every environment variable from `render.yaml`.

### Troubleshooting

| Symptom in the logs | Cause / fix |
| --- | --- |
| The build installs `torch` or `sentence-transformers` | Render is building the old code: push the new commit and use **Manual Deploy → Deploy latest commit** |
| `Ran out of memory (used over 512MB)` | Set `DRIVEDOCS_ENABLE_IMAGE_SEARCH=false`, or use a paid plan (§2) |
| `OfflineModeIsEnabled` during the build | The build command lost its `HF_HUB_OFFLINE=0` prefix; restore it from `render.yaml` |
| `429` / timeouts from huggingface.co during the build | Hub rate limit: retry, or add a read-only `HF_TOKEN` secret in the dashboard |
| `Model '…' could not be loaded` and the health check fails | The build-time model cache is missing; check `HF_HOME` is unchanged and the download step ran |
| `No open ports detected` | The start command was edited; it must contain `--host 0.0.0.0 --port $PORT` |
| Page takes ~1 minute to load | Normal on the Free plan after 15 idle minutes (the service is waking up) |
| `No matching distribution found for …` | Keep `PYTHON_VERSION=3.14.7` (the pinned wheels were verified for it) |

## 5. Verify the public website before publishing the link

From the repository folder (standard library only; waits up to 4 minutes for a sleeping service to wake):

```bash
python scripts/smoke_test.py https://<your-service>.onrender.com
```

All checks must pass (`22/22 checks passed`). Then check the UI by hand:

1. Open `https://<your-service>.onrender.com/`. The status pill turns green ("Ready").
2. Click "What is the optical transmission loss…": the answer contains **0.12 dB/cm** and a clickable
   `quantum_photonics_spec.pdf · p.1` citation, and the badge says the citations are verified.
3. Click the Mars example: "Insufficient evidence", no citations.
4. **Visual search** → "line chart of detector efficiency versus wavelength" returns the chart from page 2.
5. `https://<your-service>.onrender.com/docs` shows the API documentation.

Only then replace the placeholder in the README's **Live Demo** section with the verified URL.

## 6. Environment variables

All are set by `render.yaml`. **Nothing has to be entered manually** for the demo. Optional values:

| Variable | Where | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | Dashboard (secret, `sync: false`) | Only with `LLM_PROVIDER=gemini` / `openai`; leave empty otherwise |
| `HF_TOKEN` | Dashboard (secret), optional | Only if model downloads during the build hit Hub rate limits |
| `DRIVEDOCS_CORS_ORIGINS` | Dashboard, optional | Only if another website must call the API from the browser |

Defined in `render.yaml`: `PYTHON_VERSION=3.14.7`, `DEVICE=cpu`, `LLM_PROVIDER=extractive`,
`HF_HOME=/opt/render/project/src/.hf-cache`, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
`DRIVEDOCS_EMBEDDING_BACKEND=onnx`, `DRIVEDOCS_ENABLE_IMAGE_SEARCH=true`, `DRIVEDOCS_SEED_SAMPLE_DATA=true`,
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
