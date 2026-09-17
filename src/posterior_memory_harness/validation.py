"""Strict JSON-facing input contract for untrusted Agent adapters."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from typing import Any, Mapping


INPUT_SCHEMA_VERSION = 2
LEGACY_INPUT_SCHEMA_VERSION = 1
MAX_IDENTIFIER_LENGTH = 512
MAX_TAG_LENGTH = 128
MAX_TAGS = 64
MAX_QUERY_NODES = 256
MAX_QUERY_OBSERVATIONS = 4096
MAX_RELATION_STATES = 256
MAX_HOPS = 8
MAX_PROVENANCE_BYTES = 65_536
MAX_JSON_DEPTH = 16


class PayloadValidationError(ValueError):
    pass


OBSERVATION_V1_FIELDS = {
    "schema_version",
    "observation_id",
    "namespace",
    "relation_type",
    "source",
    "target",
    "posterior",
    "prior",
    "evidence_family",
    "provenance",
    "observed_at",
    "valid_from",
    "valid_until",
    "independent_evidence",
    "trust",
    "tags",
}

OBSERVATION_FIELDS = OBSERVATION_V1_FIELDS | {
    "source_event_id",
    "lineage_root_id",
    "derived_from",
    "encoder_revision",
    "calibration_revision",
}

QUERY_V1_FIELDS = {
    "schema_version",
    "namespace",
    "relation_type",
    "nodes",
    "seeds",
    "root",
    "as_of",
    "tags",
    "max_observations",
    "top_k",
    "hard_top1",
    "inference",
    "node_priors",
    "max_hops",
    "max_nodes",
}

QUERY_FIELDS = QUERY_V1_FIELDS | {"retrieval_policy"}

OUTCOME_FIELDS = {
    "schema_version",
    "observation_id",
    "true_relation",
    "outcome_event_id",
    "labeled_at",
    "correction_reason",
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PayloadValidationError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise PayloadValidationError(f"{field} keys must be strings")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PayloadValidationError(f"unknown fields: {', '.join(unknown)}")


def _schema_version(payload: Mapping[str, Any]) -> int:
    version = payload.get("schema_version", LEGACY_INPUT_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, Integral):
        raise PayloadValidationError("schema_version must be an integer")
    result = int(version)
    if result not in {LEGACY_INPUT_SCHEMA_VERSION, INPUT_SCHEMA_VERSION}:
        raise PayloadValidationError(
            f"unsupported input schema_version {version}; "
            f"expected {LEGACY_INPUT_SCHEMA_VERSION} or "
            f"{INPUT_SCHEMA_VERSION}"
        )
    return result


def strict_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise PayloadValidationError(f"{field} must be a string")
    if not value:
        raise PayloadValidationError(f"{field} must not be empty")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise PayloadValidationError(
            f"{field} exceeds {MAX_IDENTIFIER_LENGTH} characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PayloadValidationError(f"{field} contains control characters")
    return value


def _optional_identifier(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    return None if value is None else strict_identifier(value, field)


def strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise PayloadValidationError(f"{field} must be a boolean")
    return value


def strict_int(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PayloadValidationError(f"{field} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise PayloadValidationError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return result


def strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PayloadValidationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise PayloadValidationError(f"{field} must be finite")
    return result


def _optional_number(payload: Mapping[str, Any], field: str) -> float | None:
    value = payload.get(field)
    return None if value is None else strict_number(value, field)


def strict_string_list(
    value: Any,
    field: str,
    *,
    maximum_items: int,
    item_length: int = MAX_IDENTIFIER_LENGTH,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PayloadValidationError(f"{field} must be an array")
    if not 1 <= len(value) <= maximum_items:
        raise PayloadValidationError(
            f"{field} must contain between 1 and {maximum_items} items"
        )
    result = []
    for index, item in enumerate(value):
        text = strict_identifier(item, f"{field}[{index}]")
        if len(text) > item_length:
            raise PayloadValidationError(
                f"{field}[{index}] exceeds {item_length} characters"
            )
        result.append(text)
    if len(set(result)) != len(result):
        raise PayloadValidationError(f"{field} must not contain duplicates")
    return tuple(result)


def strict_optional_identifier_list(
    value: Any,
    field: str,
    *,
    maximum_items: int,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise PayloadValidationError(f"{field} must be an array")
    if len(value) > maximum_items:
        raise PayloadValidationError(
            f"{field} may contain at most {maximum_items} items"
        )
    result = tuple(
        strict_identifier(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise PayloadValidationError(f"{field} must not contain duplicates")
    return result


def strict_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise PayloadValidationError("tags must be an array")
    if len(value) > MAX_TAGS:
        raise PayloadValidationError(f"tags may contain at most {MAX_TAGS} items")
    result = []
    for index, item in enumerate(value):
        text = strict_identifier(item, f"tags[{index}]")
        if len(text) > MAX_TAG_LENGTH:
            raise PayloadValidationError(
                f"tags[{index}] exceeds {MAX_TAG_LENGTH} characters"
            )
        result.append(text)
    if len(set(result)) != len(result):
        raise PayloadValidationError("tags must not contain duplicates")
    return tuple(result)


def strict_probabilities(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        mapping = _mapping(value, field)
        if not mapping:
            raise PayloadValidationError(f"{field} must not be empty")
        if len(mapping) > MAX_RELATION_STATES:
            raise PayloadValidationError(
                f"{field} may contain at most {MAX_RELATION_STATES} states"
            )
        return {
            strict_identifier(key, f"{field} label"): strict_number(
                probability, f"{field}[{key!r}]"
            )
            for key, probability in mapping.items()
        }
    if not isinstance(value, (list, tuple)) or not value:
        raise PayloadValidationError(f"{field} must be a non-empty array or object")
    if len(value) > MAX_RELATION_STATES:
        raise PayloadValidationError(
            f"{field} may contain at most {MAX_RELATION_STATES} states"
        )
    return tuple(
        strict_number(probability, f"{field}[{index}]")
        for index, probability in enumerate(value)
    )


def _json_value(value: Any, field: str, depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise PayloadValidationError(
            f"{field} exceeds maximum JSON depth {MAX_JSON_DEPTH}"
        )
    if value is None or type(value) is bool or isinstance(value, str):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise PayloadValidationError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        mapping = _mapping(value, field)
        return {
            key: _json_value(item, f"{field}.{key}", depth + 1)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{field}[{index}]", depth + 1)
            for index, item in enumerate(value)
        ]
    raise PayloadValidationError(f"{field} contains a non-JSON value")


def strict_provenance(value: Any) -> dict[str, Any]:
    result = _json_value(value, "provenance")
    if not isinstance(result, dict):
        raise PayloadValidationError("provenance must be an object")
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_PROVENANCE_BYTES:
        raise PayloadValidationError(
            f"provenance exceeds {MAX_PROVENANCE_BYTES} UTF-8 bytes"
        )
    return result


def validate_observation_payload(payload: Any) -> dict[str, Any]:
    source = _mapping(payload, "observation payload")
    version = _schema_version(source)
    _reject_unknown(
        source,
        (
            OBSERVATION_V1_FIELDS
            if version == LEGACY_INPUT_SCHEMA_VERSION
            else OBSERVATION_FIELDS
        ),
    )
    required = (
        "namespace",
        "relation_type",
        "source",
        "target",
        "posterior",
        "evidence_family",
    )
    missing = [field for field in required if field not in source]
    if missing:
        raise PayloadValidationError(
            f"missing required fields: {', '.join(missing)}"
        )
    result = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "observation_id": _optional_identifier(source, "observation_id"),
        "namespace": strict_identifier(source["namespace"], "namespace"),
        "relation_type": strict_identifier(
            source["relation_type"], "relation_type"
        ),
        "source": strict_identifier(source["source"], "source"),
        "target": strict_identifier(source["target"], "target"),
        "posterior": strict_probabilities(source["posterior"], "posterior"),
        "prior": (
            None
            if source.get("prior") is None
            else strict_probabilities(source["prior"], "prior")
        ),
        "evidence_family": strict_identifier(
            source["evidence_family"], "evidence_family"
        ),
        "provenance": strict_provenance(source.get("provenance", {})),
        "observed_at": _optional_number(source, "observed_at"),
        "valid_from": _optional_number(source, "valid_from"),
        "valid_until": _optional_number(source, "valid_until"),
        "independent_evidence": strict_bool(
            source.get("independent_evidence", True),
            "independent_evidence",
        ),
        "trust": strict_number(source.get("trust", 1.0), "trust"),
        "tags": strict_tags(source.get("tags")),
        "source_event_id": _optional_identifier(
            source, "source_event_id"
        ),
        "lineage_root_id": _optional_identifier(
            source, "lineage_root_id"
        ),
        "derived_from": strict_optional_identifier_list(
            source.get("derived_from"),
            "derived_from",
            maximum_items=256,
        ),
        "encoder_revision": _optional_identifier(
            source, "encoder_revision"
        ),
        "calibration_revision": _optional_identifier(
            source, "calibration_revision"
        ),
    }
    if (
        result["observation_id"] is not None
        and result["observation_id"] in result["derived_from"]
    ):
        raise PayloadValidationError(
            "an observation cannot derive from itself"
        )
    return result


def validate_query_payload(payload: Any) -> dict[str, Any]:
    source = _mapping(payload, "query payload")
    version = _schema_version(source)
    _reject_unknown(
        source,
        (
            QUERY_V1_FIELDS
            if version == LEGACY_INPUT_SCHEMA_VERSION
            else QUERY_FIELDS
        ),
    )
    required = ("namespace", "relation_type", "root")
    missing = [field for field in required if field not in source]
    if missing:
        raise PayloadValidationError(
            f"missing required fields: {', '.join(missing)}"
        )
    has_nodes = "nodes" in source
    has_seeds = "seeds" in source
    if has_nodes == has_seeds:
        raise PayloadValidationError(
            "query must provide exactly one of nodes or seeds"
        )
    nodes_or_seeds = "nodes" if has_nodes else "seeds"
    selected = strict_string_list(
        source[nodes_or_seeds],
        nodes_or_seeds,
        maximum_items=MAX_QUERY_NODES,
    )
    root = strict_identifier(source["root"], "root")
    if root not in selected:
        raise PayloadValidationError(f"root must be included in {nodes_or_seeds}")

    node_priors_source = _mapping(source.get("node_priors", {}), "node_priors")
    if len(node_priors_source) > len(selected):
        raise PayloadValidationError(
            "node_priors may not contain more entries than selected nodes"
        )
    unknown_prior_nodes = sorted(set(node_priors_source) - set(selected))
    if unknown_prior_nodes:
        raise PayloadValidationError(
            "node_priors contains nodes outside the query: "
            + ", ".join(unknown_prior_nodes)
        )
    node_priors = {
        strict_identifier(node, "node_priors key"): strict_probabilities(
            values, f"node_priors[{node!r}]"
        )
        for node, values in node_priors_source.items()
    }
    result = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "namespace": strict_identifier(source["namespace"], "namespace"),
        "relation_type": strict_identifier(
            source["relation_type"], "relation_type"
        ),
        nodes_or_seeds: selected,
        "root": root,
        "as_of": _optional_number(source, "as_of"),
        "tags": strict_tags(source.get("tags")),
        "max_observations": strict_int(
            source.get("max_observations", 512),
            "max_observations",
            minimum=1,
            maximum=MAX_QUERY_OBSERVATIONS,
        ),
        "top_k": strict_int(
            source.get("top_k", 3),
            "top_k",
            minimum=1,
            maximum=MAX_QUERY_NODES,
        ),
        "hard_top1": strict_bool(
            source.get("hard_top1", False), "hard_top1"
        ),
        "inference": strict_identifier(
            source.get("inference", "auto"), "inference"
        ),
        "node_priors": node_priors,
        "retrieval_policy": strict_identifier(
            source.get("retrieval_policy", "recent"),
            "retrieval_policy",
        ),
    }
    if result["retrieval_policy"] not in {
        "recent",
        "coverage",
        "cycle_aware",
    }:
        raise PayloadValidationError(
            "retrieval_policy must be recent, coverage, or cycle_aware"
        )
    if has_seeds:
        result["max_hops"] = strict_int(
            source.get("max_hops", 2),
            "max_hops",
            minimum=1,
            maximum=MAX_HOPS,
        )
        result["max_nodes"] = strict_int(
            source.get("max_nodes", 32),
            "max_nodes",
            minimum=len(selected),
            maximum=MAX_QUERY_NODES,
        )
    elif "max_hops" in source or "max_nodes" in source:
        raise PayloadValidationError(
            "max_hops and max_nodes are only valid with seeds"
        )
    return result


def validate_outcome_payload(payload: Any) -> dict[str, Any]:
    """Validate a first outcome or an explicit append-only correction."""
    source = _mapping(payload, "outcome payload")
    _schema_version(source)
    _reject_unknown(source, OUTCOME_FIELDS)
    missing = [
        field
        for field in ("observation_id", "true_relation")
        if field not in source
    ]
    if missing:
        raise PayloadValidationError(
            f"missing required fields: {', '.join(missing)}"
        )
    reason = source.get("correction_reason")
    if reason is not None:
        reason = strict_identifier(reason, "correction_reason")
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "observation_id": strict_identifier(
            source["observation_id"], "observation_id"
        ),
        "true_relation": strict_identifier(
            source["true_relation"], "true_relation"
        ),
        "outcome_event_id": _optional_identifier(
            source, "outcome_event_id"
        ),
        "labeled_at": _optional_number(source, "labeled_at"),
        "correction_reason": reason,
    }
