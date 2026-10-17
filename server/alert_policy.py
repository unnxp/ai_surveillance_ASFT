"""Alert policy routing, deduplication, and cooldown management."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from common.schemas import (
    AlertPolicyRule,
    AlertRouting,
    AlertSeverity,
    CandidateType,
    EventCandidate,
    IncidentSummary,
)
from server.database import Database


@dataclass(frozen=True)
class AlertDecision:
    is_alert: bool
    severity: AlertSeverity
    routing: AlertRouting
    is_deduped: bool
    incident: IncidentSummary


class AlertPolicyEngine:
    """Evaluates candidates against versioned alert policies, managing cooldowns and incidents."""

    def __init__(self, database: Database):
        self.database = database
        # (global_person_id, candidate_type, camera_id) -> last_alert_timestamp_utc
        self._cooldown_tracker: dict[tuple[str, CandidateType, str], datetime] = {}

    def process_candidate(self, event: EventCandidate, global_person_id: str) -> AlertDecision:
        policy = self.database.get_alert_policy()
        rule = policy.rules.get(event.candidate_type)

        # Default fallback rule if type not configured
        if rule is None:
            rule = AlertPolicyRule(
                candidate_type=event.candidate_type,
                enabled=True,
                severity=AlertSeverity.MEDIUM,
                routing=AlertRouting.REVIEW_QUEUE,
                cooldown_seconds=15.0,
            )

        observed_at = event.observed_at_utc
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        else:
            observed_at = observed_at.astimezone(timezone.utc)

        # Check cooldown deduplication
        dedup_key = (global_person_id, event.candidate_type, event.camera_id)
        last_alert_time = self._cooldown_tracker.get(dedup_key)
        is_deduped = False

        if last_alert_time is not None:
            elapsed = (observed_at - last_alert_time).total_seconds()
            if 0 <= elapsed < rule.cooldown_seconds:
                is_deduped = True

        if not is_deduped and rule.enabled:
            self._cooldown_tracker[dedup_key] = observed_at

        # Correlate into Incident Timeline in database
        incident, _ = self.database.create_or_update_incident(
            global_person_id=global_person_id,
            event=event,
            severity=rule.severity,
        )

        is_alert = (
            rule.enabled
            and not is_deduped
            and rule.routing in {AlertRouting.IMMEDIATE_ALERT, AlertRouting.REVIEW_QUEUE}
        )

        return AlertDecision(
            is_alert=is_alert,
            severity=rule.severity,
            routing=rule.routing,
            is_deduped=is_deduped,
            incident=incident,
        )
