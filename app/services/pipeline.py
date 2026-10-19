import logging
from typing import Any, Dict, Optional

from app.services.classifier import assess_document_quality, detect_document_type
from app.services.extractor import run_extractor
from app.services.ingestion import ingest
from app.services.preprocess import preprocess_image
from app.services.ocr import run_ocr
from app.services.reconciler import build_final_result
from app.services.vlm import extract_vlm

logger = logging.getLogger(__name__)


def run_pipeline(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    webhook_url: str,
    reference_id: Optional[str] = None,
    doc_type_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous end-to-end pipeline orchestrator called via run_in_executor from worker.

    Steps:
    1. Ingestion: Convert raw file bytes into PIL images.
    2. Select page 1 for single-page processing; record total page count.
    3. Preprocessing: Rotation correction and resolution capping.
    4. OCR: Text and bounding-box detection via PaddleOCR.
    5. Document Quality: Assess quality tier (GOOD/DEGRADED/POOR).
    6. Classification: Detect document type (POD/DL/RC/BILTI/UNKNOWN).
    7. Extraction: First-pass regex and spatial rule extraction.
    8. VLM: Multimodal LLM extraction with document-type specific schema.
    9. Reconciliation: Merge OCR + VLM, compute triage stage.
    10. Return final structured dictionary.
    """
    logger.info(f"[{job_id}] Starting pipeline execution for file '{filename}'")

    # Step 1: Ingest
    logger.info(f"[{job_id}] Step 1: Ingesting file bytes...")
    try:
        pages = ingest(file_bytes, filename)
        logger.info(f"[{job_id}] Step 1 completed: {len(pages)} page(s) extracted")
    except Exception as e:
        logger.error(f"[{job_id}] Step 1 failed (Ingestion): {e}")
        raise

    # Step 2: Page selection
    image = pages[0]
    pages_processed = len(pages)
    logger.info(f"[{job_id}] Step 2: Processing page 1 of {pages_processed}")

    # Step 3: Preprocessing
    logger.info(f"[{job_id}] Step 3: Preprocessing image...")
    try:
        processed_image, preprocess_meta = preprocess_image(image)
        logger.info(f"[{job_id}] Step 3 completed: {preprocess_meta}")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 3 warning (Preprocessing): {e}. Using raw image.")
        processed_image = image
        preprocess_meta = {"rotation_corrected": False, "rotation_angle": 0}

    # Step 4: OCR
    logger.info(f"[{job_id}] Step 4: Running OCR...")
    try:
        ocr_result = run_ocr(processed_image)
        logger.info(f"[{job_id}] Step 4 completed: {ocr_result.get('total_blocks', 0)} blocks found")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 4 warning (OCR): {e}. Continuing with empty OCR.")
        ocr_result = {"blocks": [], "raw_text": "", "mean_confidence": 0.0, "total_blocks": 0}

    # Step 5: Quality Assessment
    logger.info(f"[{job_id}] Step 5: Assessing document quality...")
    try:
        ocr_quality = assess_document_quality(ocr_result, preprocess_meta)
        logger.info(f"[{job_id}] Step 5 completed: quality_tier={ocr_quality.get('quality_tier')}")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 5 warning (Quality): {e}.")
        ocr_quality = {
            "mean_confidence": 0.0,
            "quality_tier": "POOR",
            "low_confidence_blocks": 0,
            "trocr_fallback_count": 0,
            "rotation_corrected": False,
            "rotation_angle": 0,
            "recommendation": "MANUAL_REVIEW — Assessment failed",
        }

    # Step 6: Classification
    logger.info(f"[{job_id}] Step 6: Classifying document type...")
    try:
        raw_text = ocr_result.get("raw_text", "")
        document_type = detect_document_type(raw_text, doc_type_hint)
        logger.info(f"[{job_id}] Step 6 completed: document_type='{document_type}'")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 6 warning (Classification): {e}.")
        document_type = doc_type_hint or "UNKNOWN"

    # Step 7: Field Extraction
    logger.info(f"[{job_id}] Step 7: Extracting rule-based fields...")
    try:
        ocr_fields = run_extractor(ocr_result, processed_image)
        logger.info(f"[{job_id}] Step 7 completed: {len(ocr_fields)} fields parsed")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 7 warning (Extraction): {e}.")
        ocr_fields = {"_extractor_error": str(e)}

    # Step 8: VLM
    logger.info(f"[{job_id}] Step 8: Invoking VLM extraction for '{document_type}'...")
    try:
        vlm_result = extract_vlm(processed_image, document_type=document_type)
        logger.info(f"[{job_id}] Step 8 completed: status={vlm_result.get('status')}")
    except Exception as e:
        logger.warning(f"[{job_id}] Step 8 warning (VLM): {e}. Proceeding with empty VLM result.")
        vlm_result = {
            "status": "error",
            "available": False,
            "error": str(e),
            "structured": None,
        }

    # Step 9: Reconciliation & Final Build
    logger.info(f"[{job_id}] Step 9: Reconciling data and computing triage stage...")
    try:
        final_payload = build_final_result(
            job_id=job_id,
            reference_id=reference_id,
            ocr_result=ocr_result,
            ocr_quality=ocr_quality,
            vlm_result=vlm_result,
            ocr_fields=ocr_fields,
            preprocess_meta=preprocess_meta,
            document_type=document_type,
            pages_processed=pages_processed,
        )
        logger.info(
            f"[{job_id}] Step 9 completed: triage_stage={final_payload.get('triage_stage')}, "
            f"reason='{final_payload.get('triage_stage_reason')}'"
        )
    except Exception as e:
        logger.error(f"[{job_id}] Step 9 failed (Build Result): {e}")
        raise

    logger.info(f"[{job_id}] Document pipeline finished successfully.")
    return final_payload
