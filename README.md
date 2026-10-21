# POD Document Verification System (VLM Backend)

An asynchronous multimodal document verification and intelligence service designed for logistics and supply chain documents, including Proof of Delivery (POD / Lorry Receipts), Driving Licences (DL), Vehicle Registration Certificates (RC), and Consignment Notes (Bilti).

The system combines deterministic computer vision and OCR pipelines with state-of-the-art Vision-Language Models (Google Gemini / Local Ollama) to parse unstructured, handwritten, and stamped logistics records, cross-validate extracted entities, classify documents into operational triage stages, persist state to PostgreSQL, and dispatch results via retry-guaranteed webhooks.

---

## Architecture & Verification Pipeline

```mermaid
flowchart TD
    A[Client Upload\nPDF / JPG / PNG / WEBP] --> B[1. Document Ingestion\npypdfium2 / pdf2image fallback]
    B --> C[2. Preprocessing\nOrientation Correction & Res Capping]
    C --> D[3. Classical OCR\nPaddleOCR Engine Singleton]
    D --> E[4. Quality Tiering & Classification\nConfidence Scoring + Keyword Heuristics]
    E --> F[5. Dual-Track Information Extraction]
    F --> F1[Track A: Spatial & Regex Extractor\nBounding-Box Proximity & Rule Parsing]
    F --> F2[Track B: Vision-Language Model\nGemini 3.5 Flash / Local Ollama\nMulti-Key Failover Pool]
    F1 --> G[6. Entity Reconciler & 12-Rule Triage]
    F2 --> G
    G --> H[7. State Persistence\nPostgreSQL Store / In-Memory Fallback]
    H --> I[8. Webhook Dispatcher\nExponential Backoff Retry 2^n]
```

---

## Architectural Highlights

- **Multi-Document Intelligence**: Native schema models and domain-engineered visual reasoning prompts for Proof of Delivery (POD), Driving Licences (DL), Vehicle Registration Certificates (RC), and Consignment Notes (Bilti).
- **Automated Gemini API Key Rotation & Failover**: Resilient key-pool rotation supporting up to 4 fallback API keys (`GEMINI_API_KEY_1..4`). If a key encounters rate limits (`HTTP 429`), quota exhaustion, or authorization errors, the engine automatically and transparently switches to the next available key.
- **Candidate Model Fallback**: Primary requests target `gemini-3.5-flash`, with seamless fallback to `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-flash-latest`, and `gemini-1.5-flash`.
- **Hybrid Extraction & Cross-Validation**: Combines bounding-box spatial proximity parsing from classical OCR with visual-semantic reasoning from VLMs to minimize hallucinations and verify handwritten endorsements.
- **Deterministic 12-Rule Triage Hierarchy**: Prioritizes operations by classifying documents into decisive operational buckets (`DAMAGED`, `SHORTAGE`, `UNLOADING_CHARGES`, `MANUAL_REVIEW`, `CLEAN`).
- **Asynchronous Processing Queue**: Uploads return `202 Accepted` immediately with a unique `job_id`, delegating heavy OCR and VLM processing to an internal worker loop with backpressure controls (`QUEUE_MAX_SIZE`).
- **Resilient Webhook Delivery**: Delivers final audit payloads to external subscriber endpoints with configurable exponential backoff (`2^attempt` seconds) and failure tracking.
- **Fault-Tolerant Storage**: Dual-mode job store using PostgreSQL via `pg8000` with graceful in-memory fallback if the database connection is unreachable.

---

## Repository Structure

```text
├── app/
│   ├── api/
│   │   ├── dependencies.py      # Magic byte validation & file size constraints
│   │   └── routes/
│   │       ├── health.py        # /health probe & queue metrics
│   │       ├── jobs.py          # /api/v1/jobs/{job_id} status polling
│   │       └── upload.py        # /api/v1/upload document ingestion endpoint
│   ├── middleware/
│   │   ├── cors.py              # Cross-origin resource sharing configuration
│   │   ├── error_handler.py     # Global exception handling & RFC 7807 responses
│   │   └── request_id.py        # X-Request-ID propagation for distributed tracing
│   ├── queue/
│   │   ├── job_store.py         # Thread-safe PostgreSQL + in-memory store
│   │   ├── models.py            # Job status and metadata definitions
│   │   └── worker.py            # Bounded async worker queue & pipeline executor
│   ├── schemas/
│   │   ├── error.py             # Standardized error schemas
│   │   ├── result.py            # Reconciled document verification schemas
│   │   ├── upload.py            # Upload request/response schemas
│   │   └── webhook.py           # Webhook notification schemas
│   ├── services/
│   │   ├── classifier.py        # Document type classifier & 12-rule triage engine
│   │   ├── extractor.py         # Spatial proximity & regex field extraction
│   │   ├── ingestion.py         # PDF page rasterization & image loading
│   │   ├── ocr.py               # PaddleOCR engine singleton
│   │   ├── pipeline.py          # Synchronous 9-stage pipeline coordinator
│   │   ├── preprocess.py        # Auto-orientation & resolution normalization
│   │   ├── reconciler.py        # Cross-validation & final payload builder
│   │   ├── vlm.py               # Multimodal VLM engine & key failover rotation
│   │   └── webhook.py           # HTTP async webhook delivery with backoff
│   ├── config.py                # Pydantic BaseSettings environment configuration
│   └── main.py                  # FastAPI application entrypoint & lifespan manager
├── Dockerfile                   # Production container specification (single worker)
├── requirements.txt             # Service dependencies
├── .env.example                 # Sanitized environment template
└── README.md                    # System documentation
```

---

## Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | `str` | `""` | Primary Google Gemini API key for VLM visual auditing. |
| `GEMINI_API_KEY_1` | `str` | `""` | Fallback Gemini API key 1 (activated on quota/rate exhaustion). |
| `GEMINI_API_KEY_2` | `str` | `""` | Fallback Gemini API key 2. |
| `GEMINI_API_KEY_3` | `str` | `""` | Fallback Gemini API key 3. |
| `GEMINI_API_KEY_4` | `str` | `""` | Fallback Gemini API key 4. |
| `GEMINI_MODEL` | `str` | `gemini-3.5-flash` | Primary Gemini model identifier for vision inference. |
| `VLM_PROVIDER` | `str` | `gemini` | Multimodal provider: `gemini` or `ollama`. |
| `OLLAMA_MODEL` | `str` | `qwen2.5-vl:7b` | Model name when running local Ollama inference. |
| `MAX_FILE_SIZE_MB` | `int` | `10` | Maximum upload file size in megabytes. |
| `QUEUE_MAX_SIZE` | `int` | `50` | Maximum asynchronous processing queue capacity. |
| `WEBHOOK_RETRY_COUNT` | `int` | `3` | Maximum delivery attempts on webhook failures. |
| `WEBHOOK_TIMEOUT_SECONDS` | `int` | `10` | HTTP request timeout in seconds per webhook delivery. |
| `ALLOWED_EXTENSIONS` | `str` | `jpg,jpeg,png,webp,pdf` | Comma-separated list of permitted file extensions. |
| `DATABASE_URL` | `str` | `None` | PostgreSQL connection string (`postgresql://username:password@localhost:5432/database_name`). |

---

## Quickstart

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/jai050803/POD_verification_system_VLM_version.git
cd POD_verification_system_VLM_version

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Populate .env with your GEMINI_API_KEY and database credentials

# 5. Start service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is served at `http://localhost:8000/docs`.

### Docker Deployment

```bash
# Build Docker image
docker build -t pod-verification-vlm .

# Run container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name pod-backend \
  pod-verification-vlm
```

> **Note:** The server executes with `--workers 1` because the underlying OCR engine singleton is optimized for single-process operation per container.

---

## API Endpoints

| Method | Path | Description | Request | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health status, queue depth, job counts | None | `{"status": "ok", "queue_size": int, "jobs_total": int}` |
| `GET` | `/admin/jobs` | Diagnostic overview of all recorded jobs | None | `List[Job]` |
| `POST` | `/api/v1/upload` | Ingest document for asynchronous verification | `multipart/form-data` (`file`, `webhook_url`, `reference_id`, `doc_type_hint`) | `202 Accepted` (`UploadResponse`) |
| `GET` | `/api/v1/jobs/{job_id}` | Retrieve processing status and audit results | None | `200 OK` (`Job`) or `404 Not Found` |

---

## Webhook Delivery Specification

Upon completion of the verification pipeline, an HTTP `POST` is dispatched to the client's configured `webhook_url`.

### Success Payload (`status: "completed"`)

```json
{
  "job_id": "7b8f9a2e-4b71-482a-a92c-5df18cf85b20",
  "status": "completed",
  "reference_id": "REF-10023",
  "result": {
    "job_id": "7b8f9a2e-4b71-482a-a92c-5df18cf85b20",
    "reference_id": "REF-10023",
    "document_type": "POD",
    "triage_stage": "CLEAN",
    "triage_stage_reason": "All checks passed",
    "narrative": "Consignment Note No. CN-88219 issued by Road Express for moving 500 carton boxes from Honeywell in Bangalore to Bacardi India in Kashipur via truck NL-01-AE-8980.",
    "human_notes": "Legible receiver stamp and clear driver endorsement.",
    "consignor": {
      "name": "Honeywell Techno India Pvt Ltd",
      "location": "Bangalore",
      "identified": true
    },
    "consignee": {
      "name": "Bacardi India (P) Ltd",
      "location": "Kashipur Godown",
      "identified": true
    },
    "pod_number": "CN-88219",
    "truck_number": "NL-01-AE-8980",
    "dispatch_date": "11/10/2026",
    "dispatch_time": null,
    "delivery_date": "13/10/2026",
    "gate_timestamps": {
      "in_date": "13/10/2026",
      "in_time": "11:00 AM",
      "out_date": "13/10/2026",
      "out_time": "03:00 PM"
    },
    "from_location": "Bangalore",
    "to_location": "Kashipur",
    "quantity_billed": "500",
    "quantity_received": "500",
    "package_type": "c/b",
    "exceptions": {
      "damage": false,
      "shortage": false,
      "unloading_charges": false,
      "excess": false,
      "remarks": "Clean delivery acknowledged",
      "delivery_status": "CLEAN_DELIVERY"
    },
    "confidence": "HIGH",
    "ocr_quality_tier": "GOOD",
    "pages_processed": 1,
    "vlm_provider": "gemini",
    "vlm_model": "gemini-3.5-flash",
    "raw_ocr_text": "...",
    "field_comparisons": []
  }
}
```

### Failure Payload (`status: "failed"`)

```json
{
  "job_id": "7b8f9a2e-4b71-482a-a92c-5df18cf85b20",
  "status": "failed",
  "reference_id": "REF-10023",
  "error": "Failed to decode image: unsupported file format",
  "result": null
}
```

---

## Triage Stage Hierarchy

Documents are evaluated against a 12-rule deterministic hierarchy to determine their operational workflow:

| Priority | Triage Stage | Operational Trigger |
| :---: | :--- | :--- |
| **1** | `DAMAGED` | Cargo damage confirmed by VLM visual inspection or explicit recipient endorsement notes. |
| **2** | `SHORTAGE` | Quantity delivered is less than billed quantity, or shortage remark flagged. |
| **3** | `UNLOADING_CHARGES` | Unloading, hamali, or additional detention charges flagged on document. |
| **4** | `MANUAL_REVIEW` | Low VLM confidence, poor OCR legibility tier, unreadable critical fields, or party ambiguities. |
| **5** | `CLEAN` | High confidence, all critical fields extracted, quantities match, and clean delivery acknowledged. |

---

## Extensibility: Adding a Document Type

1. **Register Document Keywords** (`app/services/classifier.py`):
   ```python
   EWAY_KEYWORDS = ["e-way bill", "eway bill", "ewb no", "valid from", "hsn code"]
   ```
   Add the token list into `detect_document_type()`.

2. **Define VLM Extraction Prompt** (`app/services/vlm.py`):
   Create `EWAY_VLM_PROMPT` with domain-specific JSON schema rules and register it in `PROMPTS`:
   ```python
   PROMPTS["EWAY"] = EWAY_VLM_PROMPT
   ```

All upstream ingestion, preprocessing, OCR, worker execution, persistence, and webhook delivery modules will automatically support the new document type.
