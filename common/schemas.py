"""Version 1.0 gateway metadata contracts; no inference or transport dependencies."""

from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _normalize_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_normalize_utc)]


class ContractModel(BaseModel):
    """Reject unknown metadata instead of silently dropping protocol fields."""

    model_config = ConfigDict(extra="forbid")


class CandidateType(str, Enum):
    POSSIBLE_PERSON_VIOLENCE = "possible_person_violence"
    POSSIBLE_PROPERTY_CRIME = "possible_property_crime"
    UNAUTHORIZED_ZONE_ENTRY = "unauthorized_zone_entry"
    UNKNOWN_PERSON_IN_RESTRICTED_ZONE = "unknown_person_in_restricted_zone"
    TRIPWIRE_CROSSING = "tripwire_crossing"
    POSSIBLE_LOITERING = "possible_loitering"
    CROSS_CAMERA_HANDOFF = "cross_camera_handoff"


class EvidenceKind(str, Enum):
    KEYFRAME = "keyframe"
    PERSON_CROP = "person_crop"
    CLIP = "clip"


class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


class EvidenceReference(ContractModel):
    evidence_id: UUID
    kind: EvidenceKind
    relative_path: NonBlankString | None = None
    sha256: Sha256 | None = None
    size_bytes: NonNegativeInt | None = None
    captured_at_utc: UtcDatetime


class MessageEnvelope(ContractModel):
    schema_version: Literal["1.0"]
    camera_id: NonBlankString
    gateway_id: NonBlankString
    gateway_boot_id: NonBlankString
    sequence_number: NonNegativeInt
    observed_at_utc: UtcDatetime
    model_versions: dict[NonBlankString, NonBlankString]
    configuration_version: NonBlankString


class CandidateDraft(ContractModel):
    """Candidate metadata before SharedGateway attaches the message envelope."""

    candidate_type: CandidateType
    local_track_id: NonBlankString | NonNegativeInt | None = None
    global_person_id: NonBlankString | None = None
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    reason_codes: list[NonBlankString] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence")
    @classmethod
    def validate_unique_evidence_ids(
        cls, references: list[EvidenceReference]
    ) -> list[EvidenceReference]:
        if len({reference.evidence_id for reference in references}) != len(references):
            raise ValueError("Evidence IDs must be unique within an event")
        return references

    @field_validator("attributes", mode="before")
    @classmethod
    def validate_json_attributes(cls, value: Any) -> Any:
        """Validate without coercion; keep error inputs JSON-safe for API responses."""
        def check(item: Any) -> None:
            if item is None or isinstance(item, (str, bool, int)):
                return
            if isinstance(item, float) and isfinite(item):
                return
            if isinstance(item, list):
                for child in item:
                    check(child)
                return
            if isinstance(item, dict) and all(isinstance(key, str) for key in item):
                for child in item.values():
                    check(child)
                return
            raise ValueError("Attributes require JSON values with finite numbers and string keys")

        try:
            check(value)
        except ValueError as error:
            # FastAPI includes validation inputs in its 422 JSON response. A raw
            # nonfinite input would itself prevent that response from serializing.
            raise ValidationError.from_exception_data(cls.__name__, [{
                "type": "value_error", "loc": (), "input": None,
                "ctx": {"error": error},
            }]) from error
        return value


class EventCandidate(CandidateDraft, MessageEnvelope):
    """An event candidate enriched with gateway provenance and ordering."""

    event_id: UUID


class CameraHeartbeat(MessageEnvelope):
    status: CameraStatus
    capture_fps: NonNegativeFloat | None = None
    ai_fps: NonNegativeFloat | None = None
    p95_latency_ms: NonNegativeFloat | None = None
    dropped_frames: NonNegativeInt
    active_tracks: NonNegativeInt
    action_backlog: NonNegativeInt


class EventIngestAck(ContractModel):
    event_id: UUID
    accepted: bool
    duplicate: bool


class EvidenceUploadAck(ContractModel):
    evidence_id: UUID
    accepted: bool
    duplicate: bool


class HeartbeatAck(ContractModel):
    camera_id: NonBlankString
    accepted: bool


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertRouting(str, Enum):
    IMMEDIATE_ALERT = "immediate_alert"
    REVIEW_QUEUE = "review_queue"
    CONTEXT_ONLY = "context_only"
    OBSERVATION_ONLY = "observation_only"


class AlertPolicyRule(ContractModel):
    candidate_type: CandidateType
    enabled: bool = True
    severity: AlertSeverity
    routing: AlertRouting
    cooldown_seconds: NonNegativeFloat = 15.0


class AlertPolicyConfig(ContractModel):
    version: NonBlankString = "1.0"
    rules: dict[CandidateType, AlertPolicyRule]


class IncidentTimelineEntry(ContractModel):
    event_id: UUID
    camera_id: NonBlankString
    candidate_type: CandidateType
    observed_at_utc: UtcDatetime
    severity: AlertSeverity
    evidence_count: NonNegativeInt = 0


class ReviewAction(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    CONFIRM = "confirm"
    REJECT = "reject"
    RECLASSIFY = "reclassify"


class IncidentReviewRequest(ContractModel):
    action: ReviewAction
    reviewed_by: NonBlankString = "operator"
    final_severity: AlertSeverity | None = None
    reclassified_type: CandidateType | None = None
    false_positive_reason: str | None = None
    notes: str | None = None


class IncidentReviewRecord(ContractModel):
    review_id: NonBlankString
    incident_id: NonBlankString
    action: ReviewAction
    reviewed_by: NonBlankString
    reviewed_at_utc: UtcDatetime
    final_severity: AlertSeverity | None = None
    reclassified_type: CandidateType | None = None
    false_positive_reason: str | None = None
    notes: str | None = None


class IncidentSummary(ContractModel):
    incident_id: NonBlankString
    global_person_id: NonBlankString
    status: Literal["open", "acknowledged", "confirmed", "rejected", "reclassified"] = "open"
    highest_severity: AlertSeverity
    created_at_utc: UtcDatetime
    updated_at_utc: UtcDatetime
    primary_camera: NonBlankString
    timeline: list[IncidentTimelineEntry] = Field(default_factory=list)
    reviews: list[IncidentReviewRecord] = Field(default_factory=list)
