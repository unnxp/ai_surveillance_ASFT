"""Face Registry contracts; API responses deliberately exclude biometric templates."""

from math import isfinite, sqrt
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from common.schemas import ContractModel, NonBlankString, NonNegativeInt, UtcDatetime

PersonId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class ZoneGrant(ContractModel):
    camera_id: NonBlankString
    zone_id: NonBlankString


class FaceTemplateInput(ContractModel):
    model_version: NonBlankString
    vector: list[float] = Field(min_length=512, max_length=512)

    @field_validator("vector", mode="before")
    @classmethod
    def require_numeric_vector(cls, value):
        if not isinstance(value, list) or any(type(number) not in (int, float) for number in value):
            raise ValueError("Face template requires numeric values")
        return value

    @field_validator("vector")
    @classmethod
    def require_unit_vector(cls, value: list[float]) -> list[float]:
        if not all(isfinite(number) for number in value):
            raise ValueError("Face template requires finite values")
        norm = sqrt(sum(number * number for number in value))
        if not 0.99 <= norm <= 1.01:
            raise ValueError("Face template must be normalized")
        return value


class FaceRegistryUpsert(ContractModel):
    display_name: NonBlankString
    role: NonBlankString
    consent_confirmed: Literal[True]
    allowed_zones: list[ZoneGrant]
    templates: list[FaceTemplateInput] = Field(min_length=1)

    @field_validator("consent_confirmed", mode="before")
    @classmethod
    def require_boolean_consent(cls, value):
        if value is not True:
            raise ValueError("Consent must be explicitly confirmed")
        return value

    @field_validator("allowed_zones")
    @classmethod
    def unique_grants(cls, grants: list[ZoneGrant]) -> list[ZoneGrant]:
        keys = [(grant.camera_id, grant.zone_id) for grant in grants]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate camera-zone grant")
        return grants


class FaceRegistrySummary(ContractModel):
    person_id: PersonId
    display_name: NonBlankString
    role: NonBlankString
    allowed_zones: list[ZoneGrant]
    template_count: NonNegativeInt
    model_versions: list[NonBlankString]
    consent_status: Literal["active"]
    consent_recorded_at_utc: UtcDatetime
    enrolled_at_utc: UtcDatetime


class FaceMatchRequest(FaceTemplateInput):
    camera_id: NonBlankString
    zone_ids: list[NonBlankString]

    @field_validator("zone_ids")
    @classmethod
    def unique_zones(cls, zones: list[str]) -> list[str]:
        if len(zones) != len(set(zones)):
            raise ValueError("Duplicate zone ID")
        return zones


class FaceMatchResult(ContractModel):
    status: Literal["registered", "unknown", "unavailable"]
    person_id: PersonId | None = None
    display_name: NonBlankString | None = None
    similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    zone_access: dict[NonBlankString, Literal["authorized", "unauthorized"]] = Field(
        default_factory=dict
    )
    model_version: NonBlankString

    @model_validator(mode="after")
    def coherent_identity(self):
        if self.status == "registered":
            if self.person_id is None or self.display_name is None or self.similarity is None:
                raise ValueError("Registered match needs identity and similarity")
        elif self.person_id is not None or self.display_name is not None or self.zone_access:
            raise ValueError("Unmatched face cannot have registered identity or access")
        if self.status == "unavailable" and self.similarity is not None:
            raise ValueError("Unavailable match cannot have similarity")
        return self


class TopologyEdgeInput(ContractModel):
    source_camera: NonBlankString
    destination_camera: NonBlankString
    min_travel_seconds: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    max_travel_seconds: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_endpoints_and_times(self):
        if self.source_camera == self.destination_camera:
            raise ValueError("Source and destination camera cannot be identical")
        if self.max_travel_seconds <= self.min_travel_seconds:
            raise ValueError("Maximum travel time must exceed minimum travel time")
        return self


class TopologyEdge(TopologyEdgeInput):
    edge_id: int


class TopologyConfig(ContractModel):
    edges: list[TopologyEdgeInput]

    @field_validator("edges")
    @classmethod
    def unique_camera_pairs(cls, edges: list[TopologyEdgeInput]) -> list[TopologyEdgeInput]:
        pairs = [(e.source_camera, e.destination_camera) for e in edges]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate topology edge for camera pair")
        return edges


class GlobalPersonSessionSummary(ContractModel):
    global_person_id: NonBlankString
    status: Literal["active", "expired"]
    created_at_utc: UtcDatetime
    last_seen_at_utc: UtcDatetime
    associated_cameras: list[NonBlankString] = Field(default_factory=list)
    person_id: PersonId | None = None
    display_name: NonBlankString | None = None
    association_status: Literal["matched", "possible_match", "ambiguous", "unknown"]
    last_similarity: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    observation_count: NonNegativeInt = 0


class TrackletSummary(ContractModel):
    """Appearance summary of one local track; sent when ready and again when the track ends."""

    camera_id: NonBlankString
    gateway_boot_id: NonBlankString
    local_track_id: NonNegativeInt
    phase: Literal["ready", "end"]
    first_seen_utc: UtcDatetime
    last_seen_utc: UtcDatetime
    sample_count: Annotated[int, Field(ge=1, le=64)]
    model_version: NonBlankString
    appearance: list[float] = Field(min_length=64, max_length=2048)

    @field_validator("appearance")
    @classmethod
    def finite_normalized(cls, vector: list[float]) -> list[float]:
        if not all(isfinite(value) for value in vector):
            raise ValueError("Appearance vector must be finite")
        norm = sqrt(sum(value * value for value in vector))
        if not 0.98 <= norm <= 1.02:
            raise ValueError("Appearance vector must be L2-normalized")
        return vector

    @model_validator(mode="after")
    def ordered_times(self):
        if self.last_seen_utc < self.first_seen_utc:
            raise ValueError("Tracklet cannot end before it starts")
        return self


class TrackletAck(ContractModel):
    global_person_id: NonBlankString
    association_status: Literal["matched", "possible_match", "ambiguous", "unknown"]
    cross_camera_handoff: bool
    appearance_similarity: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
