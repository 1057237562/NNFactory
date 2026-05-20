"""TransformReplayEngine — replays stored preprocessing encoding maps onto raw evaluation data.

Ensures category→integer mappings are identical between training and inference
by replaying the same encoder maps that were stored during preprocessing.
"""

import hashlib
from typing import Any

from .metadata import PreprocessingMetadata


class TransformReplayEngine:
    """Replays stored encoding maps from a PreprocessingMetadata onto raw row dicts.

    Supports all 7 encoding types stored by the preprocessing pipeline:
    label, ordinal, one-hot, target, frequency, binary, hash.

    Applies Z-score normalisation after all encodings when
    ``is_normalized=True`` and ``normalization_mean`` / ``normalization_std``
    are populated.
    """

    def __init__(self, meta: PreprocessingMetadata) -> None:
        self.meta = meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _column_order(self) -> list[str]:
        """Canonical column order matching the training-time feature matrix.

        The training dataloader uses ``numeric_columns`` (falling back to
        ``feature_columns``) to build the feature matrix.  The encoder maps
        and normalisation statistics are likewise stored in that order, so
        the replay engine must use the same ordering for correct predictions.
        """
        return (
            self.meta.numeric_columns
            if self.meta.numeric_columns
            else self.meta.feature_columns
        )

    def transform_row(
        self, row: dict[str, str], unknown_strategy: str = "error"
    ) -> dict[str, float]:
        """Transform a single raw row of string values into encoded floats.

        Parameters
        ----------
        row : dict[str, str]
            A single row of raw string values keyed by column name.
        unknown_strategy : str
            How to handle categories not seen during training:

            - ``"error"`` (default)
                Raise ``ValueError`` if an unknown category is encountered.
            - ``"fallback"``
                Use a safe default (``-1.0`` for label/ordinal, ``0.0`` for
                frequency, all-zeros for one-hot, global mean for target,
                index 0 for binary).

        Returns
        -------
        dict[str, float]
            Encoded values with keys matching ``feature_columns`` in order.
        """
        result: dict[str, Any] = {}
        order = self._column_order()

        # --- 1. Label encoding ---
        self._apply_label_ordinal(
            result, row, self.meta.label_encoders, unknown_strategy
        )

        # --- 2. Ordinal encoding ---
        self._apply_label_ordinal(
            result, row, self.meta.ordinal_encoders, unknown_strategy
        )

        # --- 3. Target encoding ---
        self._apply_target(
            result, row, self.meta.target_encoders, unknown_strategy
        )

        # --- 4. Frequency encoding ---
        self._apply_frequency(
            result, row, self.meta.frequency_encoders, unknown_strategy
        )

        # --- 5. One-hot encoding ---
        self._apply_one_hot(
            result, row, self.meta.one_hot_encoders, unknown_strategy
        )

        # --- 6. Binary encoding ---
        self._apply_binary(
            result, row, self.meta.binary_encoders, unknown_strategy
        )

        # --- 7. Hash encoding ---
        self._apply_hash(result, row, self.meta.hash_encoders)

        # --- Pass-through: raw numeric columns not covered by any encoder ---
        for col in order:
            if col not in result and col in row:
                try:
                    result[col] = float(row[col])
                except (ValueError, TypeError):
                    result[col] = 0.0

        # --- Normalization ---
        self._apply_normalization(result)

        # --- Reorder to match the training-time column order exactly ---
        ordered: dict[str, float] = {}
        for col in order:
            ordered[col] = result.get(col, 0.0)

        return ordered

    # ------------------------------------------------------------------
    # Per-encoding-type helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_label_ordinal(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
        unknown_strategy: str,
    ) -> None:
        """Apply label or ordinal encoding.

        Encoder map: ``{col_name: {value_str: int_index}}``
        """
        for col, mapping in encoders.items():
            raw = row.get(col, "")
            if raw in mapping:
                result[col] = float(mapping[raw])
            elif unknown_strategy == "fallback":
                result[col] = -1.0
            else:
                raise ValueError(
                    f"Unknown category '{raw}' for column '{col}'"
                )

    @staticmethod
    def _apply_target(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
        unknown_strategy: str,
    ) -> None:
        """Apply target encoding.

        Encoder map: ``{col_name: {value_str: target_mean}}``

        Unknown categories use the global mean (average of all stored target
        means) when fallback is requested.
        """
        for col, mapping in encoders.items():
            raw = row.get(col, "")
            if raw in mapping:
                result[col] = float(mapping[raw])
            elif unknown_strategy == "fallback":
                global_mean = (
                    sum(mapping.values()) / len(mapping) if mapping else 0.0
                )
                result[col] = global_mean
            else:
                raise ValueError(
                    f"Unknown category '{raw}' for column '{col}'"
                )

    @staticmethod
    def _apply_frequency(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
        unknown_strategy: str,
    ) -> None:
        """Apply frequency encoding.

        Encoder map: ``{col_name: {value_str: frequency_float}}``

        Unknown categories always return ``0.0`` regardless of strategy.
        """
        for col, mapping in encoders.items():
            raw = row.get(col, "")
            result[col] = float(mapping.get(raw, 0.0))

    @staticmethod
    def _apply_one_hot(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
        unknown_strategy: str,
    ) -> None:
        """Apply one-hot encoding — expand original column into ``{col}_{val}`` columns.

        Encoder map: ``{col_name: {value_str: int_index}}``

        The original column name is replaced by ``N`` columns named
        ``{col}_{val}``, one per known category.  Exactly one column is ``1.0``;
        the rest are ``0.0``.
        """
        for col, mapping in encoders.items():
            raw = row.get(col, "")
            if raw in mapping:
                for val in mapping:
                    result[f"{col}_{val}"] = 1.0 if val == raw else 0.0
            elif unknown_strategy == "fallback":
                for val in mapping:
                    result[f"{col}_{val}"] = 0.0
            else:
                raise ValueError(
                    f"Unknown category '{raw}' for column '{col}'"
                )

    @staticmethod
    def _apply_binary(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
        unknown_strategy: str,
    ) -> None:
        """Apply binary encoding — expand into bit columns.

        Encoder map::

            {col_name: {"values": [str, ...], "bit_width": int}}

        The original column is replaced by ``bit_width`` columns.  The original
        column name holds bit 0 (least significant); ``{col}_bit{N}`` holds
        bit *N*.  Unknown categories map to index ``0`` on fallback.
        """
        for col, info in encoders.items():
            values: list[str] = info["values"]
            bit_width: int = info["bit_width"]
            value_to_int = {v: i for i, v in enumerate(values)}

            raw = row.get(col, "")
            if raw in value_to_int:
                int_val = value_to_int[raw]
            elif unknown_strategy == "fallback":
                int_val = 0
            else:
                raise ValueError(
                    f"Unknown category '{raw}' for column '{col}'"
                )

            binary = bin(int_val)[2:].zfill(bit_width)
            result[col] = float(binary[0])
            for i, bit in enumerate(binary[1:]):
                result[f"{col}_bit{i + 1}"] = float(bit)

    @staticmethod
    def _apply_hash(
        result: dict[str, Any],
        row: dict[str, str],
        encoders: dict[str, Any],
    ) -> None:
        """Apply hash encoding — stateless md5-based bit extraction.

        Encoder map: ``{col_name: {"n_components": int, "signed": bool}}``

        The value is computed on-the-fly from the raw string (no stored map
        lookup).  The result is a comma-separated string of ``n_components``
        bits, identical to the training-time encoding.
        """
        for col, info in encoders.items():
            n_components: int = info["n_components"]
            signed: bool = info.get("signed", False)
            raw = row.get(col, "")
            hash_val = int(hashlib.md5(raw.encode()).hexdigest(), 16)

            bits: list[str] = []
            for i in range(n_components):
                bit = (hash_val >> i) & 1
                if signed:
                    bit = bit * 2 - 1
                bits.append(str(bit))

            result[col] = ",".join(bits)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _apply_normalization(self, result: dict[str, Any]) -> None:
        """Apply Z-score normalisation when the metadata says so."""
        if not self.meta.is_normalized:
            return
        if not self.meta.normalization_mean or not self.meta.normalization_std:
            return

        order = self._column_order()
        for col, mean, std in zip(
            order,
            self.meta.normalization_mean,
            self.meta.normalization_std,
        ):
            if col not in result:
                continue
            val = result[col]
            # Skip string-valued columns (e.g. hash-encoded)
            if isinstance(val, str):
                continue
            if std == 0.0:
                result[col] = 0.0
            else:
                result[col] = (float(val) - mean) / std
