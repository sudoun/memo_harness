"""PosteriorGlue-style probabilistic memory middleware for agent harnesses."""

from .harness import PosteriorMemoryHarness
from .inference import InferenceConfig
from .laws import FiniteRelationLaw, RelationLawRegistry, cyclic_law, s3_law
from .middleware import HarnessEvent, MemoryMiddleware, ObservationEncoder
from .monitoring import (
    MONITOR_REPORT_SCHEMA_VERSION,
    CalibrationDriftMonitor,
    MonitorConfig,
    MonitorOutcomeRecord,
    MonitorPredictionRecord,
    relation_law_fingerprint,
)
from .projection import DECISION_SCHEMA_VERSION, project_memory_decision
from .retrieval import RETRIEVAL_POLICIES
from .models import (
    MemoryCapsule,
    MemoryInferenceDiagnostics,
    MemoryNeighborhood,
    MemoryNeighborhoodQuery,
    MemoryObservation,
    MemoryQuery,
    MemoryRetrievalInfo,
)
from .store import (
    InMemoryMemoryStore,
    MonitorConfigurationConflictError,
    MonitorIdempotencyConflictError,
    MonitorObservationNotTrackedError,
    MonitorOutcomeConflictError,
    SQLiteMemoryStore,
)
from .sqlite_schema import (
    CURRENT_SCHEMA_VERSION,
    SchemaMigrationError,
    SQLiteSchemaStatus,
    UnsupportedSchemaVersionError,
)
from .validation import (
    INPUT_SCHEMA_VERSION,
    PayloadValidationError,
    validate_outcome_payload,
)

__all__ = [
    "FiniteRelationLaw",
    "HarnessEvent",
    "InferenceConfig",
    "InMemoryMemoryStore",
    "MemoryCapsule",
    "MemoryInferenceDiagnostics",
    "MemoryMiddleware",
    "MemoryNeighborhood",
    "MemoryNeighborhoodQuery",
    "MemoryObservation",
    "MemoryQuery",
    "MemoryRetrievalInfo",
    "ObservationEncoder",
    "CalibrationDriftMonitor",
    "MonitorConfig",
    "MonitorOutcomeRecord",
    "MonitorPredictionRecord",
    "PosteriorMemoryHarness",
    "project_memory_decision",
    "RelationLawRegistry",
    "SQLiteMemoryStore",
    "MonitorConfigurationConflictError",
    "MonitorIdempotencyConflictError",
    "MonitorObservationNotTrackedError",
    "MonitorOutcomeConflictError",
    "SQLiteSchemaStatus",
    "SchemaMigrationError",
    "UnsupportedSchemaVersionError",
    "CURRENT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "PayloadValidationError",
    "validate_outcome_payload",
    "MONITOR_REPORT_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "RETRIEVAL_POLICIES",
    "cyclic_law",
    "s3_law",
    "relation_law_fingerprint",
]

__version__ = "0.9.0"
