# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def save_video_cv2(frames, output_path, fps=30, *, crf=18, preset="fast"):
    """Save RGB frames, using H.264 compression when FFmpeg is available.

    Args:
        frames: Sequence of RGB image arrays.
        output_path: Destination video filename.
        fps: Playback frame rate.
        crf: H.264 quality setting from 0 to 51. Higher values produce smaller,
            lower-quality videos. The default 18 preserves the previous setting;
            try 23 or 28 for stronger compression. This applies to the H.264
            re-encode; the intermediate OpenCV video is already lossy.
        preset: H.264 encoding speed/compression tradeoff, from "ultrafast" to
            "veryslow". Slower presets generally compress better at similar quality.
            The default is "fast".

    Compression settings require FFmpeg with libx264. If it is unavailable or
    encoding fails, the OpenCV MPEG-4 fallback does not apply these settings.
    """
    if not isinstance(crf, int) or isinstance(crf, bool) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer between 0 and 51")
    if preset not in (
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ):
        raise ValueError("preset must be a supported x264 speed preset (ultrafast through veryslow)")
    if not frames:
        return

    height, width, _ = frames[0].shape
    new_height = height if height % 2 == 0 else height + 1
    new_width = width if width % 2 == 0 else width + 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write raw frames with mp4v (universally supported by OpenCV on Linux).
    # Then re-encode to H.264 so VSCode / browsers can play the result.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (new_width, new_height))
        if not out.isOpened():
            print(f"[video] Failed to open VideoWriter for {tmp_path}")
            return

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if frame_bgr.shape[:2] != (new_height, new_width):
                padded = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                padded[:height, :width] = frame_bgr
                frame_bgr = padded
            out.write(frame_bgr)
        out.release()

        # Keep the OpenCV output when the optional H.264 encoder is unavailable.
        if shutil.which("ffmpeg") is None:
            os.replace(tmp_path, output_path)
            print(
                f"[video] ffmpeg unavailable; saved {len(frames)} frames as MPEG-4 (crf/preset not applied) → {output_path}"
            )
            return

        # Re-encode with libx264 for broad playback compatibility.
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-vcodec",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"[video] ffmpeg re-encode failed:\n{result.stderr}")
            # Fall back: keep the mp4v file as-is
            os.replace(tmp_path, output_path)
            return

        print(f"[video] Saved {len(frames)} frames → {output_path}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
