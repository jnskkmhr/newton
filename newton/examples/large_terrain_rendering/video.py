# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory RTX recording with the alpha examples' video controls."""

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import newton.viewer


def configure_video(args, *, fps):
    """Validate recording before scene setup and override the simulation frame limit."""
    if not math.isfinite(args.video_duration) or args.video_duration <= 0:
        raise ValueError("--video-duration must be finite and positive")
    if not math.isfinite(args.video_speed) or args.video_speed <= 0:
        raise ValueError("--video-speed must be finite and positive")
    if not args.video:
        return
    if args.viewer != "rtx":
        raise ValueError("Video recording requires --viewer rtx (optionally --headless)")
    if int(fps * args.video_speed) < 1:
        raise ValueError("--video-speed produces a video frame rate below 1 FPS")
    if Path(args.video).suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 filename for --video")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Video recording requires FFmpeg with the libx264 encoder on PATH")
    args.num_frames = max(1, math.ceil(args.video_duration * fps))


class VideoRecorder:
    """Capture one RGB frame at a time, then atomically publish the encoded MP4."""

    def __init__(self, path, *, fps, frame_limit):
        self.path = Path(path).expanduser().resolve()
        self.fps, self.frame_limit = fps, frame_limit
        self.frame_count = 0
        self.last_time = 0.0
        self.process = None
        self.stderr = None
        self.directory = None
        self.closed = False

    @property
    def complete(self):
        return self.frame_count >= self.frame_limit

    def capture(self, viewer, sim_time):
        """Record after rendering; paused redraws do not extend the video."""
        if self.closed or self.complete or sim_time <= self.last_time:
            return
        if self.directory is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.directory = tempfile.TemporaryDirectory(prefix=".terrain-video-", dir=self.path.parent)
        frame_path = Path(self.directory.name) / "frame.png"
        # Use the public RTX screenshot API, which supports both headless and windowed rendering.
        viewer.save_screenshot(str(frame_path))
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            if self.process is None:
                self.size = rgb.size
                self.stderr = tempfile.TemporaryFile()
                self.process = subprocess.Popen(
                    [
                        shutil.which("ffmpeg") or "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        f"{rgb.width}x{rgb.height}",
                        "-framerate",
                        str(self.fps),
                        "-i",
                        "pipe:0",
                        "-an",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-preset",
                        "fast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(Path(self.directory.name) / "video.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=self.stderr,
                )
            if rgb.size != self.size:
                raise ValueError("The render size changed while recording; keep the viewport size fixed")
            try:
                self.process.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                self.close()
                raise
        self.frame_count += 1
        self.last_time = sim_time
        if self.frame_count == 1 or self.frame_count % 50 == 0 or self.complete:
            print(f"Recording video: {self.frame_count}/{self.frame_limit} frames", flush=True)

    def close(self):
        """Finalize a complete or interrupted recording and report encoder failures."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                code = self.process.wait()
                self.stderr.seek(0)
                error = self.stderr.read().decode(errors="replace")
                if code != 0:
                    raise RuntimeError(f"FFmpeg failed to encode video (exit {code}): {error.strip()}")
                os.replace(Path(self.directory.name) / "video.mp4", self.path)
                print(f"Saved video: {self.path} ({self.frame_count} frames at {self.fps} FPS)", flush=True)
        finally:
            if self.stderr is not None:
                self.stderr.close()
            if self.directory is not None:
                self.directory.cleanup()


class ViewerRTXVideo(newton.viewer.ViewerRTX):
    """Attach recording to completed RTX frames, including headless renders."""

    def __init__(self, args, *, fps):
        self.video = (
            VideoRecorder(args.video, fps=int(fps * args.video_speed), frame_limit=args.num_frames)
            if args.video
            else None
        )
        self.video_time = 0.0
        super().__init__(
            width=args.width,
            height=args.height,
            headless=args.headless,
            paused=args.paused,
            num_frames=None if self.video else args.num_frames,
            environment="none",
            async_rendering=False,
        )

    def begin_frame(self, time):
        self.video_time = time
        super().begin_frame(time)

    def end_frame(self):
        super().end_frame()
        if self.video is not None:
            self.video.capture(self, self.video_time)

    def is_running(self):
        return (self.video is None or not self.video.complete) and super().is_running()

    def reset_render_history(self, time):
        """Discard cached lighting when diagnostic geometry changes, including while paused."""
        if self._rtx is not None:
            self._rtx.reset(time=time)

    def close(self):
        try:
            if self.video is not None:
                self.video.close()
        finally:
            super().close()
