"""Vision-Language Model (VLM) Extraction Layer for Multi-Document Processing.

Supported Document Types:
1. POD (Proof of Delivery / Lorry Receipt)
2. DL (Driving Licence)
3. RC (Vehicle Registration Certificate)
4. BILTI (Consignment Note / Bilti)

Supported Backends:
1. Gemini API (Default / Recommended for fast cloud inference)
2. Local Ollama (qwen2.5-vl:7b / qwen2.5-vl:3b)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Prompt engineered specifically for logistics POD documents with real-world human-like reasoning
POD_VLM_PROMPT = """You are an expert logistics document auditor specializing in real-world Indian Proof of Delivery (POD) documents, Lorry Receipts (LR), Biltis, and Consignment Notes.

REAL-WORLD INDIAN POD RULES & CONTEXT:
1. Spatial Disorganization & Messy Handwriting:
   - Drivers and warehouse clerks write anywhere on the page, often ignoring printed lines, writing in margins, or splitting information across multiple lines.
   - TRUCK / VEHICLE NUMBER: In Indian logistics, commercial truck registration numbers follow patterns like State Code (NL, KA, DL, HR, UP, UK, MH, WB, etc.) + RTO Code (2 digits) + Series (1-2 letters) + 4 digits. The truck number is frequently handwritten in the left margin or near remarks across two lines (for example, 'NL-01' on one line and 'AE-8980' or 'AF-8980' below it). Combine these into a single standard truck number: 'NL-01-AE-8980' (or 'NL01AE8980'). Do not mistake partial printed codes like 'HR01' or form headers for the actual handwritten truck number.
   - CONSIGNOR & CONSIGNEE GEOGRAPHIC REASONING:
     * Cross-reference Origin and Destination with party details to verify accuracy!
     * Origin: Look at the Booking Branch / From location (e.g. 'BMM Blr-Bommasandra' = Bangalore). The Consignor is the sender from that region (e.g., 'Honeywell Techno India' / 'Ather Mica', located in Bangalore / Bommasandra / Manyata Tech Park).
     * Destination: Look at Destination / To (e.g. 'Kashipur'). The Consignee is the receiver in that city (e.g., 'Bacardi India (P) Ltd, Kashipur Godown', confirmed by the receiver stamp).
     * Provide both the company name and city/address.
   - GATE IN-TIME & OUT-TIME (TURNAROUND TIMESTAMPS):
     * Look closely at the top right margin, remarks column, and near/inside the receiver rubber stamp for gate in/out arrival and departure timestamps. For example: 'in time 11/10 11:00 AM' and 'out time 13/10 3:00 PM' (or 'In Date: 11/10/26 Time: 11:00 AM', 'Out Date: 13/10/26 Time: 3:00 PM').
     * Extract these exact gate in and gate out dates and times.
   - PACKAGE QUANTITY & LOGISTICS ABBREVIATIONS ('c/b'):
     * In Indian logistics, 'c/b', 'cb', 'CB' strictly stands for CARTON BOXES (or cases/boxes).
     * Quantities are ALWAYS integer counts of boxes (e.g., 674 c/b, 677 c/b, 500 c/b).
     * CRITICAL: NEVER output ordinals like '6th c/b' or '1st c/b' — nobody writes '6th c/b' in logistics! If cursive handwriting resembles '6th c/b', recognize that the ascending loop/stroke of the digit 4 or 7 in cursive handwriting was misread. It is '674 c/b' (or '677 c/b').
   - DAMAGE / SHORTAGE / REMARKS:
     * Look for remarks like 'one case broken' or '1 c/b damage' or 'unloading charges'.
     * If 1 case is broken out of 674 c/b, delivery status is 'DAMAGE' with damage quantity '1 c/b'.
   - UNCERTAINTY HANDLING:
     * If handwriting is ambiguous or messy, reason carefully like an experienced logistics human auditor. State any ambiguity in uncertainty_and_confidence.human_notes. Do not guess blindly.

Return strictly a valid JSON object matching this schema:
{
  "journey_summary": "Comprehensive 1-2 sentence journey narrative including truck number, route from origin to destination, gate in-time arrival, gate out-time departure, total cartons delivered, and any damage/shortage/unloading remarks.",
  "document_type": "Consignment Note | Lorry Receipt | Proof of Delivery",
  "pod_number": "string or null",
  "truck_number": "string or null",
  "origin": "Origin city/branch",
  "destination": "Destination city/branch",
  "dispatch_date": "DD/MM/YYYY or null",
  "delivery_date": "DD/MM/YYYY or null",
  "gate_timestamps": {
    "in_date": "string or null",
    "in_time": "string or null",
    "out_date": "string or null",
    "out_time": "string or null"
  },
  "consignor": {
    "name": "string or null",
    "address_or_location": "string or null",
    "gstin": "string or null"
  },
  "consignee": {
    "name": "string or null",
    "address_or_location": "string or null",
    "stamp_text": "string or null"
  },
  "quantity_billed": "string or null",
  "quantity_received": "string or null",
  "package_type": "c/b | cartons | bags | boxes",
  "exceptions_and_remarks": {
    "excess_received": {"reported": false, "quantity": null, "details": null},
    "shortage": {"reported": false, "quantity": null, "details": null},
    "damage": {"reported": false, "quantity": null, "details": null},
    "unloading_charges": {"reported": false, "amount": null, "details": null},
    "other_remarks": "string or null",
    "delivery_status": "CLEAN_DELIVERY | SHORTAGE | DAMAGE | EXCESS | DISPUTED"
  },
  "recipient_acknowledgement": {
    "signed": true,
    "stamped": true,
    "stamp_details": "string or null",
    "receiver_name": "string or null"
  },
  "freight_details": {
    "total_freight": null,
    "advance_amount": null,
    "balance_amount": null,
    "payable_at": null
  },
  "uncertainty_and_confidence": {
    "overall_confidence": "HIGH | MEDIUM | LOW",
    "human_notes": "Auditor notes on messy handwriting, split fields, or corrections",
    "uncertain_fields": []
  },
  "unreadable_fields": [],
  "narrative": "One or two sentence journey narrative: Consignment Note No. X issued by Y for moving Z packages from [Consignor Name] in [Origin] to [Consignee Name] in [Destination], with dispatch on DD/MM/YYYY at HH:MM. Include truck number and any exception.",
  "human_notes": "Plain English auditor observation: document legibility, handwriting quality, stamp clarity, ambiguous fields, OCR noise issues, anything a human reviewer should know before approving."
}
"""

DL_VLM_PROMPT = """You are an expert document auditor specializing in Indian Driving Licence (DL) documents.
Analyze the provided driving licence image and extract the key details with high accuracy.

Return strictly a valid JSON object matching this schema:
{
  "document_type": "Driving Licence",
  "dl_number": "string or null",
  "name": "string or null",
  "father_or_husband_name": "string or null",
  "date_of_birth": "DD/MM/YYYY or null",
  "blood_group": "string or null",
  "address": "string or null",
  "issue_date": "DD/MM/YYYY or null",
  "valid_from": "DD/MM/YYYY or null",
  "valid_to": "DD/MM/YYYY or null",
  "vehicle_classes": ["MCWG", "LMV", "TRANS", ...],
  "issuing_authority": "RTO name and code or null",
  "photo_present": true,
  "signature_present": true,
  "transport_endorsement": false,
  "badge_number": "string or null",
  "overall_confidence": "HIGH | MEDIUM | LOW",
  "human_notes": "Plain English auditor observation: document legibility, photo/signature clarity, hologram, scratches, expiry, or anything a reviewer should know.",
  "narrative": "One or two sentence summary: Driving Licence No. X issued to [Name], S/D/W of [Father/Husband Name], valid from [Valid From] to [Valid To] by [Issuing Authority] for vehicle classes [Classes].",
  "uncertain_fields": [],
  "unreadable_fields": []
}
"""

RC_VLM_PROMPT = """You are an expert document auditor specializing in Indian Vehicle Registration Certificate (RC) documents.
Analyze the provided registration certificate image and extract vehicle and ownership details with high precision.

Return strictly a valid JSON object matching this schema:
{
  "document_type": "Registration Certificate",
  "registration_number": "string or null",
  "owner_name": "string or null",
  "father_or_husband_name": "string or null",
  "address": "string or null",
  "chassis_number": "string or null",
  "engine_number": "string or null",
  "maker_model": "string or null",
  "body_type": "string or null",
  "fuel_type": "PETROL | DIESEL | CNG | ELECTRIC | HYBRID | null",
  "colour": "string or null",
  "seating_capacity": "string or null",
  "unladen_weight": "string or null",
  "gross_weight": "string or null",
  "registration_date": "DD/MM/YYYY or null",
  "fitness_valid_upto": "DD/MM/YYYY or null",
  "insurance_valid_upto": "DD/MM/YYYY or null",
  "tax_valid_upto": "DD/MM/YYYY or null",
  "hypothecation": false,
  "hypothecated_to": "Bank/financier name or null",
  "issuing_rto": "RTO name and code or null",
  "state": "State name or code or null",
  "overall_confidence": "HIGH | MEDIUM | LOW",
  "human_notes": "Plain English auditor observation: document legibility, chip/seal visibility, fitness status, hypothecation status, or any unreadable text.",
  "narrative": "One or two sentence summary: Vehicle Registration No. X registered to [Owner Name] for [Maker Model] (Chassis: Y, Engine: Z), registered at [RTO, State] on [Date], fitness valid upto [Date].",
  "uncertain_fields": [],
  "unreadable_fields": []
}
"""

BILTI_VLM_PROMPT = """You are an expert logistics document auditor specializing in Indian Bilti, Lorry Receipts (LR), and Consignment Notes.
Analyze the provided bilti/consignment note image and extract all booking, freight, and cargo specifications.

Return strictly a valid JSON object matching this schema:
{
  "document_type": "Bilti / Consignment Note",
  "bilti_number": "string or null",
  "booking_date": "DD/MM/YYYY or null",
  "from_station": "string or null",
  "to_station": "string or null",
  "consignor_name": "string or null",
  "consignee_name": "string or null",
  "description_of_goods": "string or null",
  "number_of_articles": "string or null",
  "weight_actual": "string or null",
  "weight_charged": "string or null",
  "freight_amount": "string or null",
  "payment_mode": "TOPAY | PAID | TBB | null",
  "lorry_number": "string or null",
  "driver_name": "string or null",
  "remarks": "string or null",
  "overall_confidence": "HIGH | MEDIUM | LOW",
  "human_notes": "Plain English auditor observation: handwriting quality, freight payment clarity, tax/stamp details, or ambiguous quantities.",
  "narrative": "One or two sentence summary: Bilti No. X booked on [Booking Date] from [From Station] to [To Station] by [Consignor Name] for [Consignee Name], moving [Articles] articles weighing [Weight] via lorry [Lorry Number] under payment mode [Payment Mode].",
  "uncertain_fields": [],
  "unreadable_fields": []
}
"""

PROMPTS: Dict[str, str] = {
    "POD": POD_VLM_PROMPT,
    "DL": DL_VLM_PROMPT,
    "RC": RC_VLM_PROMPT,
    "BILTI": BILTI_VLM_PROMPT,
}


def _pil_to_base64_jpeg(image: Image.Image, max_dim: int = 1600, quality: int = 90) -> str:
    """Resize image if needed and convert to base64 JPEG string preserving resolution."""
    w, h = image.size
    longest = max(w, h)
    if longest > max_dim:
        scale = float(max_dim) / float(longest)
        image = image.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            resample=getattr(Image, "Resampling", Image).LANCZOS,
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Strip markdown backticks (```json ... ```) and parse JSON safely."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"Could not parse valid JSON from VLM output:\n{raw_text[:500]}")


def call_gemini_vlm(
    image: Image.Image,
    prompt: str = POD_VLM_PROMPT,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: int = 75,
) -> Dict[str, Any]:
    """Call Google Gemini Vision API using REST generateContent with structured output and automatic fallback."""
    import requests

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured in environment or .env file.")

    primary_model = model_name or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    candidate_models = []
    for m in [
        primary_model,
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    b64_image = _pil_to_base64_jpeg(image)

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": key,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}},
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    }

    last_error = None
    for cur_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"No candidates returned by Gemini: {data}")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise RuntimeError(f"Empty content parts from Gemini: {data}")

                raw_text = parts[0].get("text", "")
                parsed_json = _clean_json_response(raw_text)

                return {
                    "provider": "gemini",
                    "model": cur_model,
                    "raw_text": raw_text,
                    "structured": parsed_json,
                }
            elif resp.status_code in (429, 404, 503):
                last_error = f"HTTP {resp.status_code} ({cur_model}): {resp.text}"
                continue
            else:
                raise RuntimeError(f"Gemini API returned HTTP {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            last_error = f"Network/timeout error on {cur_model}: {str(e)}"
            continue

    raise RuntimeError(f"All Gemini candidate models failed. Last error: {last_error}")


def call_ollama_vlm(
    image: Image.Image,
    prompt: str = POD_VLM_PROMPT,
    model_name: str = "qwen2.5-vl:7b",
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call Local Ollama instance (qwen2.5-vl:7b or qwen2.5-vl:3b) via ollama python client."""
    import tempfile
    import ollama

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_img_path = tmp.name
        image.save(temp_img_path, format="PNG")

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\nOutput strictly valid JSON only.",
                    "images": [temp_img_path],
                }
            ],
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        )

        content = response["message"]["content"]
        parsed_json = _clean_json_response(content)

        return {
            "provider": "ollama",
            "model": model_name,
            "raw_text": content,
            "structured": parsed_json,
        }
    finally:
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass


def extract_vlm(
    image: Image.Image,
    document_type: str = "POD",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Run VLM document extraction on the preprocessed image.

    Args:
        image: Preprocessed PIL Image.
        document_type: Document type ('POD', 'DL', 'RC', 'BILTI'). Defaults to 'POD'.
        provider: 'gemini' or 'ollama'. If None, detects from environment.

    Returns:
        dict:
            status: 'success' | 'error' | 'skipped'
            available: bool
            provider: str
            model: str
            raw_text: str
            structured: dict
    """
    doc_type_upper = (document_type or "POD").strip().upper()
    active_prompt = PROMPTS.get(doc_type_upper, POD_VLM_PROMPT)

    active_provider = provider or os.environ.get("VLM_PROVIDER", "gemini").lower()

    # If Gemini is requested or GEMINI_API_KEY is set
    if active_provider == "gemini" or (os.environ.get("GEMINI_API_KEY") and active_provider != "ollama"):
        try:
            res = call_gemini_vlm(image, prompt=active_prompt)
            return {
                "status": "success",
                "available": True,
                **res,
            }
        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "provider": "gemini",
                "error": str(e),
                "structured": None,
            }

    # If Ollama is configured
    if active_provider == "ollama":
        try:
            model = os.environ.get("OLLAMA_MODEL", "qwen2.5-vl:7b")
            res = call_ollama_vlm(image, prompt=active_prompt, model_name=model)
            return {
                "status": "success",
                "available": True,
                **res,
            }
        except Exception as e:
            return {
                "status": "error",
                "available": False,
                "provider": "ollama",
                "error": str(e),
                "structured": None,
            }

    return {
        "status": "skipped",
        "available": False,
        "error": "No VLM provider configured. Set GEMINI_API_KEY in .env or run local Ollama.",
        "structured": None,
    }
