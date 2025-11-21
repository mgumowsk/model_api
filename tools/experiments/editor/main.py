"""Command-line video editor pipeline built on ffmpeg.

The script stitches together opening scenes, filler loops, and a banner clip
with optional crossfade transitions, and mixes the supplied voiceover with
background music. All heavy lifting happens in ffmpeg; Python coordinates the
individual steps so users can describe the desired edit declaratively.

Example usage::

	python main.py \
		--opening-scenes open1.mp4 open2.mp4 \
		--filler-scenes filler.mp4 \
		--voiceover vo.wav \
		--bg-music music.mp3 \
		--banner endcard.mp4 \
		--output final.mp4

Filler clips and background music are optional; omit them to fall back to a
black screen section or a voiceover-only mix.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def log(message: str) -> None:
	print(f"[editor] {message}")


class FFmpegError(RuntimeError):
	"""Raised when an ffmpeg child process exits with a non-zero status."""


def ensure_binaries_available() -> None:
	log("Checking ffmpeg/ffprobe availability...")
	for binary in ("ffmpeg", "ffprobe"):
		if shutil.which(binary) is None:
			raise FFmpegError(f"Required binary '{binary}' is not available in PATH.")


def run(cmd: Sequence[str]) -> None:
	log("Running: " + " ".join(cmd))
	try:
		subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
		stderr = exc.stderr.decode() if exc.stderr else ""
		raise FFmpegError(f"Command failed: {' '.join(cmd)}\n{stderr}") from exc


def ffprobe_duration(path: str, cache: Dict[str, float]) -> float:
	if path in cache:
		return cache[path]
	cmd = [
		"ffprobe",
		"-v",
		"error",
		"-show_entries",
		"format=duration",
		"-of",
		"default=noprint_wrappers=1:nokey=1",
		path,
	]
	log("Running: " + " ".join(cmd))
	try:
		result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
		stderr = exc.stderr.decode() if exc.stderr else ""
		raise FFmpegError(f"ffprobe failed for {path}: {stderr}") from exc
	duration = float(result.stdout.decode().strip())
	cache[path] = duration
	return duration


def has_video_stream(path: str) -> bool:
	cmd = [
		"ffprobe",
		"-v",
		"error",
		"-select_streams",
		"v",
		"-show_entries",
		"stream=index",
		"-of",
		"csv=p=0",
		path,
	]
	log("Running: " + " ".join(cmd))
	try:
		result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough
		stderr = exc.stderr.decode() if exc.stderr else ""
		raise FFmpegError(f"ffprobe failed for {path}: {stderr}") from exc
	return bool(result.stdout.strip())


def parse_resolution(value: str) -> Tuple[int, int]:
	try:
		width_str, height_str = value.lower().split("x", 1)
		width = int(width_str)
		height = int(height_str)
	except ValueError as exc:  # pragma: no cover - defensive
		raise argparse.ArgumentTypeError("Resolution must look like WIDTHxHEIGHT") from exc
	if width <= 0 or height <= 0:
		raise argparse.ArgumentTypeError("Resolution dimensions must be positive integers")
	return width, height


@dataclass
class PipelineConfig:
	opening_scenes: List[str]
	filler_scenes: List[str]
	banner: List[str]
	voiceover: str
	bg_music: Optional[str]
	subtitles: Optional[str]
	output: str
	crf: int
	preset: str
	width: int
	height: int
	frame_rate: int
	transition_type: str
	transition_duration: float
	music_volume: float


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Command-line video editor powered by ffmpeg")
	parser.add_argument("--opening-scenes", nargs="+", required=True, help="Opening video files")
	parser.add_argument(
		"--filler-scenes",
		nargs="*",
		default=[],
		help="Optional filler video files; falls back to a black screen if omitted",
	)
	parser.add_argument("--banner", nargs="+", required=True, help="Banner video file(s) that close the edit")
	parser.add_argument("--voiceover", required=True, help="Voiceover audio track")
	parser.add_argument("--bg-music", help="Optional background music audio track")
	parser.add_argument("--subtitles", help="Optional subtitles file to burn into the final video (e.g., .srt)")
	parser.add_argument("--output", default="output.mp4", help="Destination video path (default output.mp4)")
	parser.add_argument("--crf", type=int, default=18, help="H.264 Constant Rate Factor (quality)")
	parser.add_argument("--preset", default="medium", help="ffmpeg preset used for encoding")
	parser.add_argument(
		"--resolution",
		default="1920x1080",
		help="Output resolution in WIDTHxHEIGHT (default 1920x1080)",
	)
	parser.add_argument(
		"--frame-rate",
		type=int,
		default=30,
		help="Target frames per second (30 is generally ideal for YouTube unless footage is 60FPS)",
	)
	parser.add_argument(
		"--transition-type",
		choices=("crossfade", "cut"),
		default="crossfade",
		help="Transition between clips (crossfade = crossfade, cut = hard cuts)",
	)
	parser.add_argument(
		"--transition-duration",
		type=float,
		default=0.2,
		help="Transition length in seconds (ignored when using 'cut')",
	)
	parser.add_argument(
		"--music-volume",
		type=float,
		default=0.10,
		help="Background music gain multiplier (0.10 = 10%% volume)",
	)
	return parser


def normalize_video(
	source: str,
	label: str,
	cfg: PipelineConfig,
	cache_dir: Path,
) -> str:
	width, height = cfg.width, cfg.height
	source_path = Path(source)
	cache_dir.mkdir(parents=True, exist_ok=True)
	cached_name = f"{source_path.stem}_n_{width}x{height}.mp4"
	cached_path = cache_dir / cached_name
	if cached_path.exists() and cached_path.stat().st_size > 0:
		log(f"Reusing cached normalization for '{source}' -> {cached_path}")
		return str(cached_path)
	log(f"Normalizing clip '{source}' as {label}")
	vf_chain = ",".join(
		[
			f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease",  # preserve AR then center-pad using lanczos
			f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
			f"fps={cfg.frame_rate}",
		]
	)
	cmd = [
		"ffmpeg",
		"-y",
		"-i",
		source,
		"-vf",
		vf_chain,
		"-an",
		"-r",
		str(cfg.frame_rate),
		"-c:v",
		"libx264",
		"-preset",
		cfg.preset,
		"-crf",
		str(cfg.crf),
		"-pix_fmt",
		"yuv420p",
		str(cached_path),
	]
	run(cmd)
	return str(cached_path)


def concat_two(
	clip_a: str,
	clip_b: str,
	tmpdir: Path,
	label: str,
	cfg: PipelineConfig,
) -> str:
	out_path = tmpdir / f"concat_{label}.mp4"
	# Use concat demuxer for stream copy if possible, but here we use filter for simplicity.
	# However, user requested stream copy for cuts.
	# To do stream copy concat, we need to create a list file.
	list_file = tmpdir / f"list_{label}.txt"
	with open(list_file, "w") as f:
		f.write(f"file '{Path(clip_a).resolve()}'\n")
		f.write(f"file '{Path(clip_b).resolve()}'\n")
	
	cmd = [
		"ffmpeg",
		"-y",
		"-f", "concat",
		"-safe", "0",
		"-i", str(list_file),
		"-c", "copy",
		str(out_path),
	]
	run(cmd)
	return str(out_path)


def crossfade_two(
	clip_a: str,
	clip_b: str,
	tmpdir: Path,
	label: str,
	cfg: PipelineConfig,
	duration_cache: Dict[str, float],
) -> str:
	out_path = tmpdir / f"xfade_{label}.mp4"
	dur_a = ffprobe_duration(clip_a, duration_cache)
	dur_b = ffprobe_duration(clip_b, duration_cache)
	overlap = min(cfg.transition_duration, dur_a, dur_b)
	if overlap <= 0:
		return concat_two(clip_a, clip_b, tmpdir, label, cfg)
	offset = max(dur_a - overlap, 0)
	filter_complex = (
		"[0:v]format=yuv420p,setsar=1[v0];"
		"[1:v]format=yuv420p,setsar=1[v1];"
		f"[v0][v1]xfade=transition=fade:duration={overlap}:offset={offset}[xf];"
		"[xf]format=yuv420p[vout]"
	)
	cmd = [
		"ffmpeg",
		"-y",
		"-i",
		clip_a,
		"-i",
		clip_b,
		"-filter_complex",
		filter_complex,
		"-map",
		"[vout]",
		"-c:v",
		"libx264",
		"-preset",
		cfg.preset,
		"-crf",
		str(cfg.crf),
		"-pix_fmt",
		"yuv420p",
		str(out_path),
	]
	run(cmd)
	return str(out_path)


def combine_sequence(
	clips: Sequence[str],
	tmpdir: Path,
	label: str,
	cfg: PipelineConfig,
	duration_cache: Dict[str, float],
) -> str:
	if not clips:
		raise ValueError("At least one clip is required to combine")
	if len(clips) == 1:
		return clips[0]

	# Optimization: Use concat demuxer for hard cuts (stream copy)
	if cfg.transition_type == "cut" or cfg.transition_duration <= 0:
		out_path = tmpdir / f"combined_{label}.mp4"
		list_file = tmpdir / f"list_{label}.txt"
		with open(list_file, "w") as f:
			for clip in clips:
				# Use absolute paths to avoid issues with relative paths in concat file
				abs_path = Path(clip).resolve()
				f.write(f"file '{abs_path}'\n")
		
		cmd = [
			"ffmpeg",
			"-y",
			"-f", "concat",
			"-safe", "0",
			"-i", str(list_file),
			"-c", "copy",
			str(out_path),
		]
		run(cmd)
		return str(out_path)

	current = clips[0]
	for idx, clip in enumerate(clips[1:], start=1):
		tag = f"{label}_{idx}"
		current = crossfade_two(current, clip, tmpdir, tag, cfg, duration_cache)
	return current


def loop_clip_to_duration(
	clip: str,
	target_duration: float,
	tmpdir: Path,
	label: str,
	cfg: PipelineConfig,
	duration_cache: Dict[str, float],
) -> str:
	cycle_duration = ffprobe_duration(clip, duration_cache)
	if cycle_duration <= 0:
		raise FFmpegError(f"Clip {clip} has zero duration; cannot loop")
	# Ensure we overshoot slightly so trimming never undershoots.
	loops_needed = max(1, math.ceil((target_duration + cfg.transition_duration) / cycle_duration))
	
	trimmed_path = tmpdir / f"trimmed_{label}.mp4"
	cmd: List[str] = ["ffmpeg", "-y"]
	if loops_needed > 1:
		cmd += ["-stream_loop", str(loops_needed - 1)]
	cmd += [
		"-i",
		clip,
		"-t",
		f"{target_duration:.3f}",
		"-c",
		"copy",
		str(trimmed_path),
	]
	run(cmd)
	return str(trimmed_path)


def render_black_clip(
	duration: float,
	tmpdir: Path,
	label: str,
	cfg: PipelineConfig,
) -> str:
	out_path = tmpdir / f"black_{label}.mp4"
	cmd = [
		"ffmpeg",
		"-y",
		"-f",
		"lavfi",
		"-i",
		f"color=c=black:s={cfg.width}x{cfg.height}:r={cfg.frame_rate}",
		"-t",
		f"{duration:.3f}",
		"-c:v",
		"libx264",
		"-preset",
		cfg.preset,
		"-crf",
		str(cfg.crf),
		"-pix_fmt",
		"yuv420p",
		str(out_path),
	]
	run(cmd)
	return str(out_path)


def _escape_subtitle_path(path: str) -> str:
	# Escape characters ffmpeg treats specially inside filter arguments.
	return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def burn_subtitles(
	video: str,
	subtitles: str,
	tmpdir: Path,
	cfg: PipelineConfig,
) -> str:
	sub_path = str(Path(subtitles).expanduser())
	log(f"Burning subtitles from '{subtitles}'")
	out_path = tmpdir / "subtitled_timeline.mp4"
	cmd = [
		"ffmpeg",
		"-y",
		"-i",
		video,
		"-vf",
		f"subtitles='{_escape_subtitle_path(sub_path)}'",
		"-c:v",
		"libx264",
		"-preset",
		cfg.preset,
		"-crf",
		str(cfg.crf),
		"-pix_fmt",
		"yuv420p",
		"-c:a",
		"copy",
		str(out_path),
	]
	run(cmd)
	return str(out_path)


def mix_audio(
	video: str,
	voiceover: str,
	music: Optional[str],
	video_duration: float,
	cfg: PipelineConfig,
	output: str,
) -> None:
	duration_str = f"{video_duration:.3f}"
	if music:
		filter_complex = (
			f"[1:a]aresample=async=1,apad,atrim=0:{duration_str}[voice];"
			f"[2:a]aresample=async=1,apad,atrim=0:{duration_str},volume={cfg.music_volume}[music];"
			"[voice][music]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
		)
		cmd = [
			"ffmpeg",
			"-y",
			"-i",
			video,
			"-i",
			voiceover,
			"-i",
			music,
			"-filter_complex",
			filter_complex,
			"-map",
			"0:v",
			"-map",
			"[aout]",
			"-c:v",
			"copy",
			"-c:a",
			"aac",
			output,
		]
	else:
		filter_complex = f"[1:a]aresample=async=1,apad,atrim=0:{duration_str}[aout]"
		cmd = [
			"ffmpeg",
			"-y",
			"-i",
			video,
			"-i",
			voiceover,
			"-filter_complex",
			filter_complex,
			"-map",
			"0:v",
			"-map",
			"[aout]",
			"-c:v",
			"copy",
			"-c:a",
			"aac",
			output,
		]
	run(cmd)


def prepare_config(parsed: argparse.Namespace) -> PipelineConfig:
	width, height = parse_resolution(parsed.resolution)
	transition_duration = parsed.transition_duration
	if parsed.transition_type == "cut":
		transition_duration = 0.0
	return PipelineConfig(
		opening_scenes=parsed.opening_scenes,
		filler_scenes=parsed.filler_scenes,
		banner=parsed.banner,
		voiceover=parsed.voiceover,
		bg_music=parsed.bg_music,
		subtitles=parsed.subtitles,
		output=parsed.output,
		crf=parsed.crf,
		preset=parsed.preset,
		width=width,
		height=height,
		frame_rate=parsed.frame_rate,
		transition_type=parsed.transition_type,
		transition_duration=transition_duration,
		music_volume=parsed.music_volume,
	)


def validate_inputs(cfg: PipelineConfig) -> None:
	video_inputs = cfg.opening_scenes + cfg.filler_scenes + cfg.banner
	audio_inputs = [cfg.voiceover]
	if cfg.bg_music:
		audio_inputs.append(cfg.bg_music)
	if cfg.subtitles:
		audio_inputs.append(cfg.subtitles)
	for path in video_inputs + audio_inputs:
		if not Path(path).expanduser().exists():
			raise FileNotFoundError(f"File not found: {path}")
	for path in video_inputs:
		if not has_video_stream(path):
			raise FFmpegError(f"No video stream detected in '{path}'. Provide a valid video clip.")
	Path(cfg.output).expanduser().parent.mkdir(parents=True, exist_ok=True)



def build_opening_sequence(
	cfg: PipelineConfig,
	tmpdir: Path,
	duration_cache: Dict[str, float],
	cache_dir: Path,
) -> str:
	log(f"Building opening sequence with {len(cfg.opening_scenes)} clip(s)")
	normalized = [
		normalize_video(path, f"opening_{idx}", cfg, cache_dir)
		for idx, path in enumerate(cfg.opening_scenes)
	]
	return combine_sequence(normalized, tmpdir, "opening", cfg, duration_cache)


def build_filler_sequence(
	cfg: PipelineConfig,
	tmpdir: Path,
	duration_cache: Dict[str, float],
	target_duration: float,
	cache_dir: Path,
) -> Optional[str]:
	if target_duration <= 0:
		log("Skipping filler section (no remaining narration time)")
		return None
	log(f"Building filler loop targeting {target_duration:.2f}s")
	if not cfg.filler_scenes:
		return render_black_clip(target_duration, tmpdir, "filler", cfg)
	normalized = [
		normalize_video(path, f"filler_{idx}", cfg, cache_dir)
		for idx, path in enumerate(cfg.filler_scenes)
	]
	if len(normalized) == 1:
		cycle = normalized[0]
	else:
		cycle = combine_sequence(normalized, tmpdir, "filler_cycle", cfg, duration_cache)
	return loop_clip_to_duration(cycle, target_duration, tmpdir, "filler", cfg, duration_cache)


def build_banner_sequence(
	cfg: PipelineConfig,
	tmpdir: Path,
	duration_cache: Dict[str, float],
	cache_dir: Path,
) -> str:
	log(f"Building banner sequence with {len(cfg.banner)} clip(s)")
	normalized = [
		normalize_video(path, f"banner_{idx}", cfg, cache_dir)
		for idx, path in enumerate(cfg.banner)
	]
	return combine_sequence(normalized, tmpdir, "banner", cfg, duration_cache)



def assemble_timeline(
	opening: str,
	filler: Optional[str],
	banner: str,
	tmpdir: Path,
	cfg: PipelineConfig,
	duration_cache: Dict[str, float],
) -> str:
	clips: List[str] = [opening]
	if filler:
		clips.append(filler)
	clips.append(banner)
	return combine_sequence(clips, tmpdir, "timeline", cfg, duration_cache)


def main(argv: Sequence[str] | None = None) -> None:
	parser = build_arg_parser()
	parsed = parser.parse_args(argv)
	cfg = prepare_config(parsed)
	ensure_binaries_available()
	log("Validating inputs and probing media streams...")
	validate_inputs(cfg)

	duration_cache: Dict[str, float] = {}

	cache_dir = Path("cache")
	cache_dir.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(prefix="cli_video_editor_") as tmp:
		tmpdir = Path(tmp)
		voice_duration = ffprobe_duration(cfg.voiceover, duration_cache)
		if voice_duration <= 0:
			raise FFmpegError("Voiceover duration must be greater than zero")

		opening = build_opening_sequence(cfg, tmpdir, duration_cache, cache_dir)
		opening_duration = ffprobe_duration(opening, duration_cache)
		log(f"Opening montage duration: {opening_duration:.2f}s")
		remaining_voice = max(voice_duration - opening_duration, 0.0)
		filler_target = remaining_voice + 1.0  # keep banner one second after narration
		filler = build_filler_sequence(cfg, tmpdir, duration_cache, filler_target, cache_dir)
		banner = build_banner_sequence(cfg, tmpdir, duration_cache, cache_dir)
		video_track = assemble_timeline(opening, filler, banner, tmpdir, cfg, duration_cache)
		if cfg.subtitles:
			video_track = burn_subtitles(video_track, cfg.subtitles, tmpdir, cfg)
		video_duration = ffprobe_duration(video_track, duration_cache)
		log(f"Final video duration: {video_duration:.2f}s")
		mix_audio(video_track, cfg.voiceover, cfg.bg_music, video_duration, cfg, cfg.output)

	log(f"Video successfully rendered to {cfg.output}")


if __name__ == "__main__":
	try:
		main()
	except (FFmpegError, FileNotFoundError, ValueError) as exc:
		print(f"Error: {exc}", file=sys.stderr)
		sys.exit(1)
