#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import Any

import numpy as np

from .image_model import ImageModel
from .result import DetectedKeypoints, DetectionResult
from .types import ListValue


class KeypointDetectionModel(ImageModel):
    """A wrapper that implements a basic keypoint regression model."""

    __model__ = "keypoint_detection"

    def __init__(self, inference_adapter, configuration: dict = {}, preload=False):
        """Initializes the keypoint detection model.

        Args:
            inference_adapter (InferenceAdapter): inference adapter containing the underlying model.
            configuration (dict, optional): configuration overrides the model parameters (see parameters() method).
              Defaults to {}.
            preload (bool, optional): forces inference adapter to load the model. Defaults to False.
        """
        super().__init__(inference_adapter, configuration, preload)
        self._check_io_number(1, 2)

    def postprocess(
        self,
        outputs: dict[str, np.ndarray],
        meta: dict[str, Any],
    ) -> DetectedKeypoints:
        """Applies SCC decoded to the model outputs.

        Args:
            outputs (dict[str, np.ndarray]): raw outputs of the model
            meta (dict[str, Any]): meta information about the input data

        Returns:
            DetectedKeypoints: detected keypoints
        """
        encoded_kps = list(outputs.values())
        batch_keypoints, batch_scores = _decode_simcc(*encoded_kps)
        orig_h, orig_w = meta["original_shape"][:2]
        kp_scale_h = orig_h / self.h
        kp_scale_w = orig_w / self.w
        batch_keypoints = batch_keypoints.squeeze() * np.array([kp_scale_w, kp_scale_h])
        return DetectedKeypoints(batch_keypoints, batch_scores.squeeze())

    @classmethod
    def parameters(cls) -> dict:
        parameters = super().parameters()
        parameters.update(
            {
                "labels": ListValue(
                    description="List of class labels",
                    value_type=str,
                    default_value=[],
                ),
            },
        )
        return parameters


class TopDownKeypointDetectionPipeline:
    """Pipeline implementing top down keypoint detection approach."""

    def __init__(self, base_model: KeypointDetectionModel) -> None:
        self.base_model = base_model

    def predict(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
    ) -> list[DetectedKeypoints]:
        """Predicts keypoints for the given image and detections.

        Args:
            image (np.ndarray): input full-size image
            detection_result (detection_result): detections located within the given image

        Returns:
            list[DetectedKeypoints]: per detection keypoints in detection coordinates
        """
        crops = []
        for box in detection_result.bboxes:
            x1, y1, x2, y2 = box
            crops.append(image[y1:y2, x1:x2])

        crops_results = self.predict_crops(crops)
        for i, box in enumerate(detection_result.bboxes):
            x1, y1, x2, y2 = box
            crops_results[i] = DetectedKeypoints(
                crops_results[i].keypoints + np.array([x1, y1]),
                crops_results[i].scores,
            )

        return crops_results

    def predict_crops(self, crops: list[np.ndarray]) -> list[DetectedKeypoints]:
        """Predicts keypoints for the given crops.

        Args:
            crops (list[np.ndarray]): list of cropped object images

        Returns:
            list[DetectedKeypoints]: per crop keypoints
        """
        return self.base_model.infer_batch(crops)


def _decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decodes keypoint coordinates from SimCC representations. The decoded coordinates are in the input image space.

    Args:
        simcc_x (np.ndarray): SimCC label for x-axis
        simcc_y (np.ndarray): SimCC label for y-axis
        simcc_split_ratio (float): The ratio of the label size to the input size.

    Returns:
        tuple:
        - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
        - scores (np.ndarray): The keypoint scores in shape (N, K).
            It usually represents the confidence of the keypoint prediction
    """
    keypoints, scores = _get_simcc_maximum(simcc_x, simcc_y)

    # Unsqueeze the instance dimension for single-instance results
    if keypoints.ndim == 2:
        keypoints = keypoints[None, :]
        scores = scores[None, :]

    keypoints /= simcc_split_ratio

    return keypoints, scores


def _get_simcc_maximum(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Hy) or (N, K, Hy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    if simcc_x.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_x.shape}"
        raise ValueError(msg)
    if simcc_y.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_y.shape}"
        raise ValueError(msg)
    if simcc_x.ndim != simcc_y.ndim:
        msg = f"{simcc_x.shape} != {simcc_y.shape}"
        raise ValueError(msg)

    if simcc_x.ndim == 3:
        batch_size, num_keypoints, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(batch_size * num_keypoints, -1)
        simcc_y = simcc_y.reshape(batch_size * num_keypoints, -1)
    else:
        batch_size = None

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.take_along_axis(
        simcc_x,
        np.expand_dims(x_locs, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)
    max_val_y = np.take_along_axis(
        simcc_y,
        np.expand_dims(y_locs, axis=-1),
        axis=-1,
    ).squeeze(axis=-1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    if batch_size:
        locs = locs.reshape(batch_size, num_keypoints, 2)
        vals = vals.reshape(batch_size, num_keypoints)

    return locs, vals
