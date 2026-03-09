"""Inference-only predictor for robust/sklearn logistic regression weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Predictor:
    """Loads pre-trained weights and runs binary predictions (-1 / +1)."""

    _SUPPORTED_MODELS = ("robust", "sklearn")
    _REQUIRED_KEYS = (
        "robust_w",
        "robust_b",
        "sklearn_w",
        "sklearn_b",
        "scaler_min",
        "scaler_scale",
    )

    def __init__(self, weights_path: str | Path = "weights.npz", model_type: str = "robust"):
        self.weights_path = Path(weights_path)
        self.model_type = model_type

        if self.model_type not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Expected one of: {', '.join(self._SUPPORTED_MODELS)}."
            )

        self._loaded = False
        self._w: np.ndarray | None = None
        self._b: float | None = None
        self._scaler_min: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._n_features: int | None = None

    def _check_loaded(self) -> None:
        if (
            self._w is None
            or self._b is None
            or self._scaler_min is None
            or self._scaler_scale is None
            or self._n_features is None
        ):
            raise RuntimeError("Internal error: predictor weights were not loaded correctly.")

    def _validate_input(self, data: np.ndarray) -> np.ndarray:
        X = np.asarray(data, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                "`data` must be a 2D array with shape (n_samples, n_features). "
                f"Got array with ndim={X.ndim}."
            )
        return X

    def _raw_linear_params(self) -> tuple[np.ndarray, float]:
        self._check_loaded()
        w = self._w
        b = self._b
        scaler_min = self._scaler_min
        scaler_scale = self._scaler_scale
        if w is None or b is None or scaler_min is None or scaler_scale is None:
            raise RuntimeError("Internal error: model parameters are unavailable.")

        # score = (X * scale + min) @ w + b = X @ (w * scale) + (min @ w + b)
        raw_w = w * scaler_scale
        raw_b = float(scaler_min @ w + b)
        return raw_w, raw_b

    def _load_weights_once(self) -> None:
        if self._loaded:
            return

        with np.load(self.weights_path, allow_pickle=False) as weights:
            missing = [key for key in self._REQUIRED_KEYS if key not in weights.files]
            if missing:
                raise KeyError(
                    "Missing required keys in weights file "
                    f"'{self.weights_path}': {', '.join(missing)}."
                )

            w_key = f"{self.model_type}_w"
            b_key = f"{self.model_type}_b"

            w = np.asarray(weights[w_key], dtype=float).reshape(-1)
            b_arr = np.asarray(weights[b_key], dtype=float).reshape(-1)
            scaler_min = np.asarray(weights["scaler_min"], dtype=float).reshape(-1)
            scaler_scale = np.asarray(weights["scaler_scale"], dtype=float).reshape(-1)

        if w.size == 0:
            raise ValueError(f"'{w_key}' is empty in '{self.weights_path}'.")
        if b_arr.size == 0:
            raise ValueError(f"'{b_key}' is empty in '{self.weights_path}'.")
        if scaler_min.size == 0 or scaler_scale.size == 0:
            raise ValueError(
                f"'scaler_min' and 'scaler_scale' must be non-empty in '{self.weights_path}'."
            )
        if scaler_min.shape != scaler_scale.shape:
            raise ValueError(
                "Scaler parameter shape mismatch: "
                f"scaler_min={scaler_min.shape}, scaler_scale={scaler_scale.shape}."
            )
        if w.shape != scaler_min.shape:
            raise ValueError(
                "Weights and scaler parameter shape mismatch: "
                f"{w_key}={w.shape}, scaler_min={scaler_min.shape}."
            )

        self._w = w
        self._b = float(b_arr[0])
        self._scaler_min = scaler_min
        self._scaler_scale = scaler_scale
        self._n_features = w.shape[0]
        self._loaded = True

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Compute raw decision scores for a 2D batch of samples."""
        X = self._validate_input(data)
        self._load_weights_once()
        self._check_loaded()
        w = self._w
        b = self._b
        scaler_min = self._scaler_min
        scaler_scale = self._scaler_scale
        n_features = self._n_features
        if (
            w is None
            or b is None
            or scaler_min is None
            or scaler_scale is None
            or n_features is None
        ):
            raise RuntimeError("Internal error: model parameters are unavailable.")

        if X.shape[1] != n_features:
            raise ValueError(
                "Feature size mismatch: "
                f"expected {n_features}, got {X.shape[1]}."
            )

        X_scaled = X * scaler_scale + scaler_min
        return X_scaled @ w + b

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict class labels for a 2D batch of samples."""
        scores = self.decision_function(data)
        return np.where(scores >= 0, 1, -1).astype(np.int64)

    def shap_values(
        self,
        data: np.ndarray,
        observation_index: int,
        baseline: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Exact SHAP values for the linear decision function."""
        X = self._validate_input(data)
        self._load_weights_once()
        self._check_loaded()
        n_features = self._n_features
        if n_features is None:
            raise RuntimeError("Internal error: number of features is unavailable.")

        if X.shape[1] != n_features:
            raise ValueError(
                "Feature size mismatch: "
                f"expected {n_features}, got {X.shape[1]}."
            )
        if not (0 <= observation_index < X.shape[0]):
            raise ValueError(
                "observation_index is out of range: "
                f"expected 0 <= index < {X.shape[0]}, got {observation_index}."
            )

        if baseline is None:
            baseline_arr = X.mean(axis=0)
        else:
            baseline_arr = np.asarray(baseline, dtype=float).reshape(-1)
            if baseline_arr.shape[0] != n_features:
                raise ValueError(
                    "Baseline feature size mismatch: "
                    f"expected {n_features}, got {baseline_arr.shape[0]}."
                )

        raw_w, raw_b = self._raw_linear_params()
        x_obs = X[observation_index]

        shap_vals = (x_obs - baseline_arr) * raw_w
        base_value = float(raw_b + baseline_arr @ raw_w)
        score = float(raw_b + x_obs @ raw_w)

        return {
            "base_value": base_value,
            "score": score,
            "shap_values": shap_vals,
            "baseline": baseline_arr,
            "observation": x_obs,
        }
