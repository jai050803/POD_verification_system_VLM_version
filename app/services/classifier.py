from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

DELIVERY_KEYWORDS: List[str] = [
    "delivery",
    "delivered",
    "consignee",
    "consignor",
    "lorry",
    "receipt",
    "pod",
    "proof",
    "vehicle",
    "truck",
    "freight",
    "dispatch",
    "shipment",
    "cargo",
    "goods",
    "transport",
    "gr no",
    "lr no",
    "waybill",
    "invoice",
    "quantity",
]

# Additional keyword lists for document types
DL_KEYWORDS: List[str] = [
    "driving licence",
    "driving license",
    "dl no",
    "licence no",
    "date of birth",
    "blood group",
    "valid upto",
    "transport",
    "non-transport",
    "badge no",
    "rto",
    "motor vehicle",
]

RC_KEYWORDS: List[str] = [
    "registration certificate",
    "rc no",
    "registration no",
    "chassis no",
    "engine no",
    "fuel type",
    "seating capacity",
    "unladen weight",
    "body type",
    "maker",
    "model",
    "colour",
    "fitness upto",
]

BILTI_KEYWORDS: List[str] = [
    "bilti",
    "bility",
    "consignment note",
    "gr no",
    "lr no",
    "booking date",
    "freight",
    "consignor",
    "consignee",
    "to pay",
    "paid",
    "article",
    "weight",
    "charged weight",
]


def is_delivery_document(raw_text: str) -> Dict[str, Any]:
    """Determine if a document is a valid delivery or logistics document.

    Checks raw OCR text for delivery and transport keywords.

    Args:
        raw_text: Full raw OCR text string.

    Returns:
        dict:
            is_delivery_document: bool (True if matched_count >= 3)
            matched_keywords: List[str]
            matched_count: int
    """
    text_lower = (raw_text or "").lower()
    matched_keywords: List[str] = [
        kw for kw in DELIVERY_KEYWORDS if kw in text_lower
    ]
    matched_count = len(matched_keywords)
    is_delivery = matched_count >= 3

    return {
        "is_delivery_document": is_delivery,
        "matched_keywords": matched_keywords,
        "matched_count": matched_count,
    }


def detect_document_type(raw_text: str, doc_type_hint: Optional[str] = None) -> str:
    """
    Detect document type from hint or keyword scoring.
    - If doc_type_hint is provided and is one of POD/DL/RC/BILTI, returns it directly.
    - Otherwise scores each type by keyword matches in raw_text (case-insensitive).
    - Minimum 2 keyword matches required; else returns "UNKNOWN".
    - POD wins ties.
    Returns: "POD" | "DL" | "RC" | "BILTI" | "UNKNOWN"
    """
    if doc_type_hint:
        hint_clean = doc_type_hint.strip().upper()
        if hint_clean in ("POD", "DL", "RC", "BILTI"):
            return hint_clean

    text_lower = (raw_text or "").lower()

    pod_score = sum(1 for kw in DELIVERY_KEYWORDS if kw in text_lower)
    dl_score = sum(1 for kw in DL_KEYWORDS if kw in text_lower)
    rc_score = sum(1 for kw in RC_KEYWORDS if kw in text_lower)
    bilti_score = sum(1 for kw in BILTI_KEYWORDS if kw in text_lower)

    scores = {
        "POD": pod_score,
        "DL": dl_score,
        "RC": rc_score,
        "BILTI": bilti_score,
    }

    max_score = max(scores.values())
    if max_score < 2:
        return "UNKNOWN"

    # POD wins ties
    if scores["POD"] == max_score:
        return "POD"

    # Highest score wins among other types
    for doc_type in ["DL", "RC", "BILTI"]:
        if scores[doc_type] == max_score:
            return doc_type

    return "UNKNOWN"


def assess_document_quality(
    ocr_result: Dict[str, Any], preprocess_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess document OCR quality and generate review recommendations.

    Args:
        ocr_result: Dictionary containing 'mean_confidence', 'blocks', etc.
        preprocess_meta: Metadata from preprocessing steps (rotation, stamps).

    Returns:
        Flat dictionary containing quality metrics, counts, and recommendation.
    """
    mean_confidence = float(ocr_result.get("mean_confidence", 0.0))
    blocks = ocr_result.get("blocks", [])

    # Quality tier assignment
    if mean_confidence >= 0.75:
        quality_tier = "GOOD"
    elif mean_confidence >= 0.50:
        quality_tier = "DEGRADED"
    else:
        quality_tier = "POOR"

    # Count block metrics
    low_confidence_blocks = sum(
        1 for b in blocks if float(b.get("confidence", 0.0)) < 0.75
    )
    trocr_fallback_count = sum(
        1 for b in blocks if b.get("method") == "TROCR_FALLBACK"
    )

    # Preprocessing metadata extraction
    rotation_corrected = bool(preprocess_meta.get("rotation_corrected", False))
    rotation_angle = int(preprocess_meta.get("rotation_angle", 0))

    # Recommendation determination
    if quality_tier == "POOR":
        recommendation = (
            "MANUAL_REVIEW — OCR confidence too low to trust extraction"
        )
    elif trocr_fallback_count > 5:
        recommendation = (
            "VERIFY — Multiple handwritten fields detected, review flagged fields"
        )
    else:
        recommendation = (
            "AUTO_APPROVED — Document quality sufficient for automated processing"
        )

    return {
        "mean_confidence": round(mean_confidence, 4),
        "quality_tier": quality_tier,
        "low_confidence_blocks": low_confidence_blocks,
        "trocr_fallback_count": trocr_fallback_count,
        "rotation_corrected": rotation_corrected,
        "rotation_angle": rotation_angle,
        "recommendation": recommendation,
    }


def _parse_quantity_int(val: Any) -> Optional[int]:
    """Helper to parse quantity fields into an integer."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "n/a"):
        return None
    # Remove commas
    cleaned = s.replace(",", "")
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        pass
    # Regex leading digits: e.g. "10 boxes" -> 10
    match = re.search(r"^\s*(\d+)", cleaned)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            pass
    return None


def compute_triage_stage(
    vlm_structured: Dict[str, Any], ocr_quality: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Computes the triage stage and reason for a processed document.
    Priority order: DAMAGED, SHORTAGE, UNLOADING_CHARGES, MANUAL_REVIEW, CLEAN

    Rules (first match wins):
    1. damage.reported=True OR delivery_status in (DAMAGE) -> DAMAGED, reason="Damage reported"
    2. shortage.reported=True OR delivery_status=SHORTAGE -> SHORTAGE, reason="Shortage reported"
    3. unloading_charges.reported=True -> UNLOADING_CHARGES, reason="Unloading charges flagged"
    4. delivery_status in (DISPUTED, EXCESS) -> MANUAL_REVIEW, reason="Delivery status disputed or excess"
    5. quality_tier=POOR -> MANUAL_REVIEW, reason="OCR quality too poor to trust"
    6. overall_confidence=LOW -> MANUAL_REVIEW, reason="VLM confidence too low"
    7. Any critical fields in uncertain_fields -> MANUAL_REVIEW, reason="Critical field uncertain: {field}"
    8. quantity_billed and quantity_received both present and NOT equal (parse as int)
       -> MANUAL_REVIEW, reason="Quantity mismatch: billed {x} != received {y}"
    9. Cannot parse quantity fields as int -> MANUAL_REVIEW, reason="Quantity unreadable"
    10. overall_confidence=MEDIUM AND uncertain_fields non-empty
        -> MANUAL_REVIEW, reason="Medium confidence with uncertain fields"
    11. overall_confidence=HIGH AND quality_tier in (GOOD, DEGRADED) -> CLEAN, reason="All checks passed"
    12. Default -> MANUAL_REVIEW, reason="Insufficient confidence for clean approval"
    """
    vlm_data = vlm_structured or {}
    ocr_data = ocr_quality or {}

    exceptions = vlm_data.get("exceptions", {}) or {}
    damage_info = exceptions.get("damage") or vlm_data.get("damage") or {}
    shortage_info = exceptions.get("shortage") or vlm_data.get("shortage") or {}
    unloading_info = exceptions.get("unloading_charges") or vlm_data.get("unloading_charges") or {}

    delivery_status = str(vlm_data.get("delivery_status", "")).strip().upper()

    # 1. Damage check
    damage_reported = bool(damage_info.get("reported", False)) if isinstance(damage_info, dict) else False
    if damage_reported or delivery_status in ("DAMAGE", "DAMAGED"):
        return "DAMAGED", "Damage reported"

    # 2. Shortage check
    shortage_reported = bool(shortage_info.get("reported", False)) if isinstance(shortage_info, dict) else False
    if shortage_reported or delivery_status == "SHORTAGE":
        return "SHORTAGE", "Shortage reported"

    # 3. Unloading charges check
    unloading_reported = bool(unloading_info.get("reported", False)) if isinstance(unloading_info, dict) else False
    if unloading_reported:
        return "UNLOADING_CHARGES", "Unloading charges flagged"

    # 4. Delivery status disputed or excess
    if delivery_status in ("DISPUTED", "EXCESS"):
        return "MANUAL_REVIEW", "Delivery status disputed or excess"

    # 5. Quality tier POOR
    quality_tier = str(ocr_data.get("quality_tier", "")).strip().upper()
    if quality_tier == "POOR":
        return "MANUAL_REVIEW", "OCR quality too poor to trust"

    # 6. Overall confidence LOW
    confidence = (
        vlm_data.get("overall_confidence")
        or vlm_data.get("confidence")
        or vlm_data.get("uncertainty_and_confidence", {}).get("overall_confidence", "")
    )
    overall_confidence = str(confidence).strip().upper() if confidence else ""
    if overall_confidence == "LOW":
        return "MANUAL_REVIEW", "VLM confidence too low"

    # 7. Critical fields in uncertain_fields
    uncertain_fields = (
        vlm_data.get("uncertain_fields")
        or vlm_data.get("uncertainty_and_confidence", {}).get("uncertain_fields")
        or []
    )
    critical_fields = [
        "quantity_received",
        "quantity_billed",
        "truck_number",
        "consignee",
        "delivery_date",
    ]
    for crit in critical_fields:
        if crit in uncertain_fields:
            return "MANUAL_REVIEW", f"Critical field uncertain: {crit}"

    # 8 & 9. Quantity parsing and comparison
    qty_billed_raw = vlm_data.get("quantity_billed")
    qty_received_raw = vlm_data.get("quantity_received")

    has_billed = qty_billed_raw is not None and str(qty_billed_raw).strip() != "" and str(qty_billed_raw).strip().lower() not in ("none", "null")
    has_received = qty_received_raw is not None and str(qty_received_raw).strip() != "" and str(qty_received_raw).strip().lower() not in ("none", "null")

    if has_billed and has_received:
        qty_billed_int = _parse_quantity_int(qty_billed_raw)
        qty_received_int = _parse_quantity_int(qty_received_raw)

        if qty_billed_int is None or qty_received_int is None:
            return "MANUAL_REVIEW", "Quantity unreadable"

        if qty_billed_int != qty_received_int:
            return "MANUAL_REVIEW", f"Quantity mismatch: billed {qty_billed_int} != received {qty_received_int}"
    elif has_billed:
        if _parse_quantity_int(qty_billed_raw) is None:
            return "MANUAL_REVIEW", "Quantity unreadable"
    elif has_received:
        if _parse_quantity_int(qty_received_raw) is None:
            return "MANUAL_REVIEW", "Quantity unreadable"

    # 10. Medium confidence with uncertain fields
    if overall_confidence == "MEDIUM" and len(uncertain_fields) > 0:
        return "MANUAL_REVIEW", "Medium confidence with uncertain fields"

    # 11. High confidence and acceptable quality tier
    if overall_confidence == "HIGH" and quality_tier in ("GOOD", "DEGRADED"):
        return "CLEAN", "All checks passed"

    # 12. Default fallback
    return "MANUAL_REVIEW", "Insufficient confidence for clean approval"
