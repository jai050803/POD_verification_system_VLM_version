"""Reconciliation and Comparative Fusion Layer for POD OCR & VLM.

Merges traditional OCR output (PaddleOCR) with Vision-Language Model
(Gemini / Qwen2.5-VL) extraction to produce the highest-confidence,
audit-proof structured logistics result.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize_text(val: Any) -> str:
    """Normalize text for fuzzy string comparison."""
    if val is None:
        return ""
    s = str(val).strip().upper()
    return re.sub(r"[\s\-_.,/]+", "", s)


def _clean_val(val: Any) -> Optional[str]:
    """Return clean stripped string or None if unreadable / empty."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() in ("NONE", "NULL", "UNREADABLE", "UNKNOWN", "NOT_FOUND"):
        return None
    return s


def _extract_party_str(val: Any) -> Optional[str]:
    """Format company party name and optional city/location."""
    if isinstance(val, dict):
        name = _clean_val(val.get("name"))
        addr = _clean_val(val.get("address_or_location"))
        if name and addr:
            return f"{name} ({addr})"
        return name or addr
    return _clean_val(val)


def reconcile(
    ocr_fields: Dict[str, Any],
    vlm_result: Dict[str, Any],
    ocr_raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reconcile traditional OCR fields with VLM output.

    Args:
        ocr_fields: Structured dictionary from pipeline.extractor.extract_fields()
        vlm_result: Full dictionary from pipeline.vlm.extract_vlm()
        ocr_raw: Raw OCR results dictionary containing raw_text and blocks

    Returns:
        Dictionary containing:
        - journey_summary: 1-2 sentence narrative journey description
        - delivery_status: Overall delivery assessment
        - reconciled_fields: Final merged high-confidence fields
        - field_comparisons: Side-by-side audit comparison for each field
        - exceptions_and_remarks: Detailed breakdown of excess, shortage, damage, unloading
        - vlm_metadata: Provider, model, availability info
    """
    vlm_available = vlm_result.get("available", False) and bool(vlm_result.get("structured"))
    vlm_data: Dict[str, Any] = vlm_result.get("structured") or {}

    field_comparisons: List[Dict[str, Any]] = []

    def add_comparison(
        field_key: str,
        display_name: str,
        ocr_val: Any,
        vlm_val: Any,
        ocr_conf: Optional[str] = None,
        prefer_vlm_handwriting: bool = False,
    ) -> Any:
        clean_ocr = _clean_val(ocr_val)
        clean_vlm = _clean_val(vlm_val)

        norm_ocr = _normalize_text(clean_ocr)
        norm_vlm = _normalize_text(clean_vlm)

        final_val = None
        agreement = "UNKNOWN"
        source = "NONE"
        notes = ""

        if not vlm_available:
            final_val = clean_ocr
            agreement = "OCR_ONLY"
            source = "OCR"
            notes = "VLM was unavailable; field sourced exclusively from OCR."
        elif norm_ocr and norm_vlm and (norm_ocr == norm_vlm or norm_ocr in norm_vlm or norm_vlm in norm_ocr):
            # Both agree
            final_val = clean_vlm if len(str(clean_vlm)) >= len(str(clean_ocr)) else clean_ocr
            agreement = "AGREED"
            source = "BOTH"
            notes = "Both OCR and VLM converged on this field with high confidence."
        elif clean_vlm and not clean_ocr:
            # VLM captured what OCR missed (e.g. handwriting, stamps, faint ink)
            final_val = clean_vlm
            agreement = "VLM_ENHANCED"
            source = "VLM"
            notes = "VLM identified text obscured or missed by standard OCR."
        elif clean_ocr and not clean_vlm:
            # OCR found it via strict label regex
            final_val = clean_ocr
            agreement = "OCR_PREFERRED"
            source = "OCR"
            notes = "OCR captured printed field near label."
        elif clean_vlm and clean_ocr:
            # Discrepancy between OCR and VLM
            if prefer_vlm_handwriting:
                final_val = clean_vlm
                agreement = "VLM_RESOLVED"
                source = "VLM"
                notes = "VLM contextual understanding preferred for handwriting / stamps."
            elif ocr_conf == "HIGH" and len(norm_ocr) >= 6:
                final_val = clean_ocr
                agreement = "OCR_PREFERRED"
                source = "OCR"
                notes = "OCR high confidence printed regex match prioritized."
            else:
                final_val = clean_vlm
                agreement = "VLM_RESOLVED"
                source = "VLM"
                notes = "VLM preferred after resolving character ambiguity."
        else:
            final_val = None
            agreement = "NOT_FOUND"
            source = "NONE"
            notes = "Field could not be identified by either engine."

        field_comparisons.append({
            "field_key": field_key,
            "display_name": display_name,
            "ocr_value": clean_ocr,
            "ocr_confidence": ocr_conf,
            "vlm_value": clean_vlm,
            "final_value": final_val,
            "agreement": agreement,
            "source": source,
            "notes": notes,
        })

        return final_val

    # 1. Truck / Vehicle Number
    ocr_veh = ocr_fields.get("vehicle_number", {})
    ocr_veh_val = ocr_veh.get("value") if isinstance(ocr_veh, dict) else ocr_fields.get("truck_number")
    ocr_veh_conf = ocr_veh.get("confidence") if isinstance(ocr_veh, dict) else None
    vlm_veh_val = vlm_data.get("truck_number")
    final_truck = add_comparison("truck_number", "Truck / Vehicle No.", ocr_veh_val, vlm_veh_val, ocr_veh_conf)

    # 2. POD / LR Number
    ocr_pod = ocr_fields.get("pod_number", {})
    ocr_pod_val = ocr_pod.get("value") if isinstance(ocr_pod, dict) else ocr_pod
    ocr_pod_conf = ocr_pod.get("confidence") if isinstance(ocr_pod, dict) else None
    vlm_pod_val = vlm_data.get("pod_number")
    final_pod = add_comparison("pod_number", "POD / LR Number", ocr_pod_val, vlm_pod_val, ocr_pod_conf)

    # 3. Delivery Date
    ocr_date = ocr_fields.get("date", {})
    ocr_date_val = ocr_date.get("value") if isinstance(ocr_date, dict) else ocr_fields.get("delivery_date")
    ocr_date_conf = ocr_date.get("confidence") if isinstance(ocr_date, dict) else None
    vlm_date_val = vlm_data.get("delivery_date") or vlm_data.get("dispatch_date")
    final_date = add_comparison("delivery_date", "Delivery Date", ocr_date_val, vlm_date_val, ocr_date_conf)

    # 4. Route: From / Origin
    ocr_route = ocr_fields.get("route", {})
    ocr_from_val = ocr_route.get("from_location") if isinstance(ocr_route, dict) else ocr_fields.get("from_location")
    vlm_from_val = vlm_data.get("origin")
    final_from = add_comparison("from_location", "Origin (From)", ocr_from_val, vlm_from_val, prefer_vlm_handwriting=True)

    # 5. Route: To / Destination
    ocr_to_val = ocr_route.get("to_location") if isinstance(ocr_route, dict) else ocr_fields.get("to_location")
    vlm_to_val = vlm_data.get("destination")
    final_to = add_comparison("to_location", "Destination (To)", ocr_to_val, vlm_to_val, prefer_vlm_handwriting=True)

    # 6. Consignor (Sender)
    ocr_consignor = ocr_fields.get("consignor", {})
    ocr_consignor_val = ocr_consignor.get("value") if isinstance(ocr_consignor, dict) else None
    vlm_consignor_raw = vlm_data.get("consignor")
    vlm_consignor_val = _extract_party_str(vlm_consignor_raw)
    final_consignor = add_comparison("consignor", "Consignor (Sender)", ocr_consignor_val, vlm_consignor_val, prefer_vlm_handwriting=True)

    # 7. Consignee (Receiver / Customer)
    ocr_consignee = ocr_fields.get("consignee", {})
    ocr_consignee_val = ocr_consignee.get("value") if isinstance(ocr_consignee, dict) else ocr_fields.get("customer_name")
    vlm_consignee_raw = vlm_data.get("consignee")
    vlm_consignee_val = _extract_party_str(vlm_consignee_raw)
    final_consignee = add_comparison("consignee", "Consignee (Customer)", ocr_consignee_val, vlm_consignee_val, prefer_vlm_handwriting=True)

    # 8. Quantity
    ocr_qty = ocr_fields.get("quantity", {})
    ocr_qty_val = ocr_qty.get("value") if isinstance(ocr_qty, dict) else ocr_qty
    vlm_qty_val = vlm_data.get("quantity_received") or vlm_data.get("quantity_billed")
    final_quantity = add_comparison("quantity", "Quantity / Packages", ocr_qty_val, vlm_qty_val, prefer_vlm_handwriting=True)

    # 8b. Gate In / Out Timestamps (Turnaround & Detention Tracking)
    gate_ts = vlm_data.get("gate_timestamps", {}) if isinstance(vlm_data.get("gate_timestamps"), dict) else {}
    in_str = None
    if gate_ts.get("in_date") or gate_ts.get("in_time"):
        in_str = f"{gate_ts.get('in_date', '')} {gate_ts.get('in_time', '')}".strip()
    out_str = None
    if gate_ts.get("out_date") or gate_ts.get("out_time"):
        out_str = f"{gate_ts.get('out_date', '')} {gate_ts.get('out_time', '')}".strip()

    if in_str:
        add_comparison("gate_in_time", "Gate In-Time (Arrival)", None, in_str, prefer_vlm_handwriting=True)
    if out_str:
        add_comparison("gate_out_time", "Gate Out-Time (Departure)", None, out_str, prefer_vlm_handwriting=True)

    # 9. Document Type
    ocr_doctype = ocr_fields.get("document_type", {})
    ocr_doctype_val = ocr_doctype.get("value") if isinstance(ocr_doctype, dict) else ocr_doctype
    vlm_doctype_val = vlm_data.get("document_type")
    final_doctype = add_comparison("document_type", "Document Type", ocr_doctype_val, vlm_doctype_val)

    # 10. Remarks and Exceptions (Deep Extraction)
    ocr_remarks = ocr_fields.get("remarks", {})
    ocr_remarks_val = ocr_remarks.get("value") if isinstance(ocr_remarks, dict) else ocr_remarks
    vlm_exceptions = vlm_data.get("exceptions_and_remarks", {}) if isinstance(vlm_data.get("exceptions_and_remarks"), dict) else {}
    vlm_remarks_text = vlm_exceptions.get("other_remarks")

    # Merge remarks text
    combined_remarks_list = []
    if _clean_val(vlm_remarks_text):
        combined_remarks_list.append(str(vlm_remarks_text))
    if _clean_val(ocr_remarks_val) and str(ocr_remarks_val) not in combined_remarks_list:
        combined_remarks_list.append(str(ocr_remarks_val))

    final_remarks_text = " | ".join(combined_remarks_list) if combined_remarks_list else None
    add_comparison("remarks", "Remarks & Notes", ocr_remarks_val, vlm_remarks_text, prefer_vlm_handwriting=True)

    # Parse specific logistics exceptions
    excess_info = vlm_exceptions.get("excess_received", {}) if isinstance(vlm_exceptions.get("excess_received"), dict) else {}
    shortage_info = vlm_exceptions.get("shortage", {}) if isinstance(vlm_exceptions.get("shortage"), dict) else {}
    damage_info = vlm_exceptions.get("damage", {}) if isinstance(vlm_exceptions.get("damage"), dict) else {}
    unloading_info = vlm_exceptions.get("unloading_charges", {}) if isinstance(vlm_exceptions.get("unloading_charges"), dict) else {}

    # Check OCR raw text as fallback for unloading / shortage / damage if VLM missed it
    raw_ocr_str = (ocr_raw.get("raw_text", "") if ocr_raw else "").lower()
    if not unloading_info.get("reported") and ("unloading" in raw_ocr_str or "hamali" in raw_ocr_str):
        m = re.search(r"(?:unloading|hamali|labour)[\s\w.:-]*?([0-9]+(?:[,.][0-9]+)?)", raw_ocr_str)
        if m:
            unloading_info = {
                "reported": True,
                "amount": m.group(1),
                "details": f"Detected in OCR text: {m.group(0)}",
            }

    if not shortage_info.get("reported") and ("short" in raw_ocr_str or "shortage" in raw_ocr_str):
        shortage_info = {
            "reported": True,
            "quantity": None,
            "details": "Shortage keywords found in document text",
        }

    if not damage_info.get("reported") and any(w in raw_ocr_str for w in ("damage", "broken", "leak", "torn", "wet")):
        damage_info = {
            "reported": True,
            "quantity": None,
            "details": "Damage keywords found in document text",
        }

    # Synthesize Delivery Status
    status_str = vlm_exceptions.get("delivery_status")
    if not status_str or status_str == "UNREADABLE":
        if shortage_info.get("reported") and damage_info.get("reported"):
            status_str = "SHORTAGE_AND_DAMAGE"
        elif shortage_info.get("reported"):
            status_str = "SHORTAGE"
        elif damage_info.get("reported"):
            status_str = "DAMAGE"
        elif excess_info.get("reported"):
            status_str = "EXCESS"
        elif unloading_info.get("reported"):
            status_str = "DELIVERED_WITH_UNLOADING_CHARGES"
        elif final_truck or final_pod:
            status_str = "CLEAN_DELIVERY"
        else:
            status_str = "DELIVERED"

    # Journey Narrative Summary
    journey_summary = vlm_data.get("journey_summary")
    if not journey_summary:
        truck_part = f"Truck {final_truck}" if final_truck else "Vehicle"
        from_part = f"from {final_from}" if final_from else ""
        to_part = f"to {final_to}" if final_to else ""
        route_part = f"running {from_part} {to_part}".strip()
        date_part = f"on {final_date}" if final_date else ""
        time_part = f" (Gate In: {in_str}, Gate Out: {out_str})" if (in_str or out_str) else ""
        remarks_part = f"Remarks: {final_remarks_text}" if final_remarks_text else "Clean delivery acknowledged."
        journey_summary = f"{truck_part} {route_part} arrived at destination {date_part}{time_part}. {remarks_part}".strip()

    # Freight Details
    vlm_freight = vlm_data.get("freight_details", {}) if isinstance(vlm_data.get("freight_details"), dict) else {}
    ocr_freight = ocr_fields.get("freight_details", {}) if isinstance(ocr_fields.get("freight_details"), dict) else {}

    reconciled_freight = {
        "total_freight": _clean_val(vlm_freight.get("total_freight")) or _clean_val(ocr_freight.get("total_freight")),
        "advance_amount": _clean_val(vlm_freight.get("advance_amount")) or _clean_val(ocr_freight.get("advance_amount")),
        "balance_amount": _clean_val(vlm_freight.get("balance_amount")) or _clean_val(ocr_freight.get("balance_amount")),
        "payable_at": _clean_val(vlm_freight.get("payable_at")) or _clean_val(ocr_freight.get("payable_at")),
    }

    # Recipient Acknowledgement
    recipient_ack = vlm_data.get("recipient_acknowledgement", {}) if isinstance(vlm_data.get("recipient_acknowledgement"), dict) else {}

    # Assemble Reconciled Payload
    reconciled_fields = {
        "document_type": final_doctype,
        "truck_number": final_truck,
        "vehicle_number": final_truck,
        "pod_number": final_pod,
        "delivery_date": final_date,
        "from_location": final_from,
        "to_location": final_to,
        "consignor": final_consignor,
        "consignor_details": vlm_data.get("consignor") if isinstance(vlm_data.get("consignor"), dict) else None,
        "consignee": final_consignee,
        "consignee_details": vlm_data.get("consignee") if isinstance(vlm_data.get("consignee"), dict) else None,
        "customer_name": final_consignee,
        "gate_in_time": in_str,
        "gate_out_time": out_str,
        "gate_timestamps": gate_ts,
        "quantity": final_quantity,
        "package_type": vlm_data.get("package_type"),
        "remarks": final_remarks_text,
        "delivery_status": status_str,
        "freight_details": reconciled_freight,
    }

    exceptions_breakdown = {
        "delivery_status": status_str,
        "excess_received": excess_info,
        "shortage": shortage_info,
        "damage": damage_info,
        "unloading_charges": unloading_info,
        "gate_timestamps": gate_ts,
        "other_remarks": final_remarks_text,
        "recipient_acknowledgement": recipient_ack,
        "unreadable_fields": vlm_data.get("unreadable_fields", []),
    }

    uncertainty = vlm_data.get("uncertainty_and_confidence", {})

    return {
        "journey_summary": journey_summary,
        "delivery_status": status_str,
        "reconciled_fields": reconciled_fields,
        "exceptions_and_remarks": exceptions_breakdown,
        "field_comparisons": field_comparisons,
        "uncertainty_and_confidence": uncertainty,
        "vlm_metadata": {
            "provider": vlm_result.get("provider", "none"),
            "model": vlm_result.get("model", "none"),
            "available": vlm_available,
            "status": vlm_result.get("status", "skipped"),
        },
    }


def build_final_result(
    job_id: str,
    reference_id: Optional[str],
    ocr_result: dict,
    ocr_quality: dict,
    vlm_result: dict,
    ocr_fields: dict,
    preprocess_meta: dict,
    document_type: str,
    pages_processed: int,
) -> dict:
    """
    Builds the final reconciled verification payload conforming to the PODResult schema.
    Incorporates triage classification, deep exception tracking, and party resolution.
    """
    reconciled = reconcile(ocr_fields, vlm_result, ocr_raw=ocr_result)
    vlm_data = vlm_result.get("structured") or {}

    from app.services.classifier import compute_triage_stage

    triage_stage, triage_reason = compute_triage_stage(vlm_data, ocr_quality)

    rec = reconciled.get("reconciled_fields", {})
    exceptions = reconciled.get("exceptions_and_remarks", {})

    consignor_raw = vlm_data.get("consignor", {})
    consignee_raw = vlm_data.get("consignee", {})

    damage_info = exceptions.get("damage") if isinstance(exceptions.get("damage"), dict) else {}
    shortage_info = exceptions.get("shortage") if isinstance(exceptions.get("shortage"), dict) else {}
    unloading_info = exceptions.get("unloading_charges") if isinstance(exceptions.get("unloading_charges"), dict) else {}
    excess_info = exceptions.get("excess_received") if isinstance(exceptions.get("excess_received"), dict) else {}

    return {
        "job_id": job_id,
        "reference_id": reference_id,
        "document_type": document_type,
        "triage_stage": triage_stage,
        "triage_stage_reason": triage_reason,
        "narrative": vlm_data.get("narrative") or reconciled.get("journey_summary"),
        "human_notes": vlm_data.get("human_notes"),
        "consignor": {
            "name": consignor_raw.get("name") if isinstance(consignor_raw, dict) else rec.get("consignor"),
            "location": consignor_raw.get("address_or_location") if isinstance(consignor_raw, dict) else None,
            "identified": bool(rec.get("consignor")),
        },
        "consignee": {
            "name": consignee_raw.get("name") if isinstance(consignee_raw, dict) else rec.get("consignee"),
            "location": consignee_raw.get("address_or_location") if isinstance(consignee_raw, dict) else None,
            "identified": bool(rec.get("consignee")),
        },
        "pod_number": rec.get("pod_number"),
        "truck_number": rec.get("truck_number"),
        "dispatch_date": vlm_data.get("dispatch_date"),
        "dispatch_time": None,
        "delivery_date": rec.get("delivery_date"),
        "gate_timestamps": rec.get("gate_timestamps"),
        "from_location": rec.get("from_location"),
        "to_location": rec.get("to_location"),
        "quantity_billed": vlm_data.get("quantity_billed"),
        "quantity_received": vlm_data.get("quantity_received"),
        "package_type": vlm_data.get("package_type"),
        "exceptions": {
            "damage": damage_info.get("reported", False),
            "shortage": shortage_info.get("reported", False),
            "unloading_charges": unloading_info.get("reported", False),
            "excess": excess_info.get("reported", False),
            "remarks": exceptions.get("other_remarks"),
            "delivery_status": reconciled.get("delivery_status"),
        },
        "confidence": (vlm_data.get("uncertainty_and_confidence") or {}).get("overall_confidence"),
        "ocr_quality_tier": ocr_quality.get("quality_tier"),
        "pages_processed": pages_processed,
        "vlm_provider": vlm_result.get("provider"),
        "vlm_model": vlm_result.get("model"),
        "raw_ocr_text": ocr_result.get("raw_text", ""),
        "field_comparisons": reconciled.get("field_comparisons", []),
    }
