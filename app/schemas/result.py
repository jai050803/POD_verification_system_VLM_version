from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExceptionDetail(BaseModel):
    reported: bool = False
    quantity: Optional[str] = None
    details: Optional[str] = None


class FreightDetails(BaseModel):
    total_freight: Optional[str] = None
    advance_amount: Optional[str] = None
    balance_amount: Optional[str] = None
    payable_at: Optional[str] = None


class GateTimestamps(BaseModel):
    in_date: Optional[str] = None
    in_time: Optional[str] = None
    out_date: Optional[str] = None
    out_time: Optional[str] = None


class PartyDetail(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    identified: bool = False


class TriageStage(str, Enum):
    DAMAGED = "DAMAGED"
    SHORTAGE = "SHORTAGE"
    UNLOADING_CHARGES = "UNLOADING_CHARGES"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    CLEAN = "CLEAN"


class PODResult(BaseModel):
    job_id: str
    reference_id: Optional[str] = None
    triage_stage: TriageStage = TriageStage.CLEAN
    document_type: Optional[str] = None
    narrative: Optional[str] = None
    human_notes: Optional[str] = None
    consignor: Optional[PartyDetail] = None
    consignee: Optional[PartyDetail] = None
    pod_number: Optional[str] = None
    truck_number: Optional[str] = None
    dispatch_date: Optional[str] = None
    dispatch_time: Optional[str] = None
    delivery_date: Optional[str] = None
    gate_timestamps: Optional[GateTimestamps] = None
    freight_details: Optional[FreightDetails] = None
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    quantity_billed: Optional[str] = None
    quantity_received: Optional[str] = None
    package_type: Optional[str] = None
    exceptions: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[str] = None
    ocr_quality_tier: Optional[str] = None
    pages_processed: int = 0
    vlm_provider: Optional[str] = None
    vlm_model: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    field_comparisons: List[Any] = Field(default_factory=list)
    triage_stage_reason: Optional[str] = None
