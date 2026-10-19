# POD Document Verification System (VLM Backend)

An asynchronous multimodal document verification and intelligence service designed for Indian logistics documents (Proof of Delivery / POD, Driving Licences / DL, Vehicle Registration Certificates / RC, and Biltis / Consignment Notes). The system ingests PDFs and images, executes rotation correction and resolution capping, extracts textual metadata with PaddleOCR and regex rules, performs deep semantic verification using Vision-Language Models (Google Gemini / Local Ollama), classifies the document into triage stages, persists state in PostgreSQL, and delivers structured audit results via retry-guaranteed webhooks.

---

## Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | `str` | `""` | Google Gemini API Key for VLM visual auditing. |
| `GEMINI_MODEL` | `str` | `gemini-2.0-flash` | Primary Gemini model identifier for vision inference. |
| `VLM_PROVIDER` | `str` | `gemini` | Multimodal provider: `gemini` or `ollama`. |
| `OLLAMA_MODEL` | `str` | `qwen2.5-vl:7b` | Model name when running local Ollama inference. |
| `MAX_FILE_SIZE_MB` | `int` | `10` | Maximum upload file size in megabytes. |
| `QUEUE_MAX_SIZE` | `int` | `50` | Maximum asynchronous processing queue capacity. |
| `WEBHOOK_RETRY_COUNT` | `int` | `3` | Maximum delivery attempts on webhook failures. |
| `WEBHOOK_TIMEOUT_SECONDS` | `int` | `10` | HTTP request timeout in seconds per webhook delivery. |
| `ALLOWED_EXTENSIONS` | `str` | `jpg,jpeg,png,webp,pdf` | Comma-separated list of permitted file extensions. |
| `DATABASE_URL` | `str` | `postgresql://postgres:2301420045@localhost:5432/pod_verification` | PostgreSQL database connection string. |

---

## How to Run Locally

### 1. Prerequisites
- Python 3.11+
- PostgreSQL database running with the database created (`pod_verification`)

### 2. Installation & Server Startup
```bash
# Clone the repository and enter directory
git clone https://github.com/jai050803/POD_verification_system_VLM_version.git
cd POD_verification_system_VLM_version

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and database credentials

# Start FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at `http://localhost:8000/docs`.

---

## How to Run with Docker

```bash
# Build the Docker image
docker build -t pod-verification-vlm .

# Run the container with environment variables
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
| `GET` | `/health` | Service health status, queue depth, total jobs | None | `{"status": "ok", "queue_size": int, "jobs_total": int}` |
| `GET` | `/admin/jobs` | Debug list of all stored jobs | None | `List[Job]` |
| `POST` | `/api/v1/upload` | Upload document for async processing | `multipart/form-data` (`file`, `webhook_url`, `reference_id`, `doc_type_hint`) | `202 Accepted` (`UploadResponse`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll processing status & results | None | `200 OK` (`Job` model) or `404 Not Found` |

---

## Webhook Payload Example

When background processing finishes, the service posts a payload to the client's `webhook_url`:

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
    "vlm_model": "gemini-2.0-flash",
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

## Triage Stage Definitions

| Triage Stage | Definition |
| :--- | :--- |
| `DAMAGED` | Cargo damage confirmed by VLM visual inspection or explicit recipient damage notes. |
| `SHORTAGE` | Quantity delivered is less than billed quantity, or shortage remark flagged. |
| `UNLOADING_CHARGES` | Unloading, hamali, or extra labor charges indicated on document. |
| `MANUAL_REVIEW` | Any uncertainty, low VLM confidence, poor OCR quality, unreadable critical field, or quantity mismatch. |
| `CLEAN` | High confidence, all critical fields extracted, quantities match, and no exceptions flagged. |

---

## Adding a New Document Type

To add support for a new document type (e.g. `EWAY_BILL`):

1. **Add Keyword Dictionary to Classifier** (`app/services/classifier.py`):
   ```python
   EWAY_KEYWORDS = ["e-way bill", "eway bill", "ewb no", "valid from", "hsn code"]
   ```
   Include the new list in `detect_document_type()` scoring.

2. **Add VLM Prompt Schema** (`app/services/vlm.py`):
   Define `EWAY_VLM_PROMPT` with the desired JSON output schema and register it in `PROMPTS`:
   ```python
   PROMPTS["EWAY"] = EWAY_VLM_PROMPT
   ```

3. **Done**:
   The ingestion, preprocessing, OCR, VLM orchestrator, and webhook delivery will automatically recognize and handle the new document type.
