"""Pluggable finite relation laws."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class FiniteRelationLaw:
    name: str
    labels: tuple[str, ...]
    table: np.ndarray
    inverse: np.ndarray
    identity: int = 0

    def __post_init__(self) -> None:
        size = len(self.labels)
        table = np.asarray(self.table, dtype=int)
        inverse = np.asarray(self.inverse, dtype=int)
        if (
            not isinstance(self.name, str)
            or not self.name
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in self.name
            )
        ):
            raise ValueError("law name must be a non-empty safe string")
        if (
            not self.labels
            or any(
                not isinstance(label, str)
                or not label
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in label
                )
                for label in self.labels
            )
            or len(set(self.labels)) != size
        ):
            raise ValueError("law name and unique labels are required")
        if table.shape != (size, size) or inverse.shape != (size,):
            raise ValueError("invalid table or inverse shape")
        if not 0 <= self.identity < size:
            raise ValueError("identity index is out of range")
        if np.any(inverse < 0) or np.any(inverse >= size):
            raise ValueError("inverse table contains an invalid value")
        if np.any(table < 0) or np.any(table >= size):
            raise ValueError("composition table contains an invalid value")
        for value in range(size):
            if table[self.identity, value] != value or table[value, self.identity] != value:
                raise ValueError("declared identity is invalid")
            if (
                table[value, inverse[value]] != self.identity
                or table[inverse[value], value] != self.identity
            ):
                raise ValueError("invalid inverse")
        for left in range(size):
            for middle in range(size):
                for right in range(size):
                    if table[table[left, middle], right] != table[left, table[middle, right]]:
                        raise ValueError("composition table is not associative")
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "inverse", inverse)

    @property
    def size(self) -> int:
        return len(self.labels)

    def compose(self, left: int, right: int) -> int:
        return int(self.table[int(left), int(right)])

    def relative(self, source_state: int, target_state: int) -> int:
        return self.compose(int(self.inverse[int(source_state)]), int(target_state))

    def encode_probabilities(
        self, values: Iterable[float] | dict[str, float]
    ) -> tuple[float, ...]:
        if isinstance(values, dict):
            unknown = set(values) - set(self.labels)
            if unknown:
                raise ValueError(f"unknown relation labels: {sorted(unknown)}")
            return tuple(float(values.get(label, 0.0)) for label in self.labels)
        return tuple(float(value) for value in values)

    def opposite(self, name: str | None = None) -> "FiniteRelationLaw":
        return FiniteRelationLaw(
            name=name or f"{self.name}_opposite",
            labels=self.labels,
            table=self.table.T.copy(),
            inverse=self.inverse.copy(),
            identity=self.identity,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "labels": list(self.labels),
            "table": self.table.tolist(),
            "inverse": self.inverse.tolist(),
            "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FiniteRelationLaw":
        return cls(
            name=str(payload["name"]),
            labels=tuple(str(value) for value in payload["labels"]),
            table=np.asarray(payload["table"], dtype=int),
            inverse=np.asarray(payload["inverse"], dtype=int),
            identity=int(payload.get("identity", 0)),
        )


class RelationLawRegistry:
    def __init__(self):
        self._laws: dict[str, FiniteRelationLaw] = {}

    def validate_registration(
        self, relation_type: str, law: FiniteRelationLaw
    ) -> None:
        if (
            not isinstance(relation_type, str)
            or not relation_type
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in relation_type
            )
        ):
            raise ValueError("relation_type must be a non-empty safe string")
        existing = self._laws.get(relation_type)
        if existing is not None:
            same = (
                existing.name == law.name
                and existing.labels == law.labels
                and existing.identity == law.identity
                and np.array_equal(existing.table, law.table)
                and np.array_equal(existing.inverse, law.inverse)
            )
            if not same:
                raise ValueError(
                    f"relation law for {relation_type!r} is already registered"
                )

    def register(self, relation_type: str, law: FiniteRelationLaw) -> None:
        self.validate_registration(relation_type, law)
        existing = self._laws.get(relation_type)
        if existing is not None:
            return
        self._laws[relation_type] = law

    def get(self, relation_type: str) -> FiniteRelationLaw:
        try:
            return self._laws[relation_type]
        except KeyError as error:
            raise KeyError(f"no relation law registered for {relation_type!r}") from error

    def relation_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._laws))


def cyclic_law(order: int) -> FiniteRelationLaw:
    if order < 1:
        raise ValueError("cyclic order must be positive")
    table = np.fromfunction(
        lambda left, right: (left + right) % order,
        (order, order),
        dtype=int,
    ).astype(int)
    inverse = np.asarray([(-value) % order for value in range(order)], dtype=int)
    return FiniteRelationLaw(
        name=f"C{order}",
        labels=tuple(str(value) for value in range(order)),
        table=table,
        inverse=inverse,
    )


def s3_law() -> FiniteRelationLaw:
    identity = (0, 1, 2)
    values = [identity] + [value for value in permutations(range(3)) if value != identity]
    lookup = {value: index for index, value in enumerate(values)}

    def compose(left, right):
        return tuple(left[right[index]] for index in range(3))

    table = np.empty((6, 6), dtype=int)
    for left_index, left in enumerate(values):
        for right_index, right in enumerate(values):
            table[left_index, right_index] = lookup[compose(left, right)]
    inverse = np.empty(6, dtype=int)
    for index in range(6):
        inverse[index] = int(np.flatnonzero(table[index] == 0)[0])
    labels = tuple("".join(str(value + 1) for value in permutation) for permutation in values)
    return FiniteRelationLaw("S3", labels, table, inverse)
