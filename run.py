#!/usr/bin/env python3
"""Single entry point for every stage of the traffic flow analysis.

    python run.py detect      # YOLO + ByteTrack over all 254 clips  (the slow step)
    python run.py params      # the five traffic parameters
    python run.py train       # congestion classifier + evaluation
    python run.py visuals     # figures and annotated videos
    python run.py report      # written insights and recommendations
    python run.py serve       # the website: live analysis and whole-dataset results
    python run.py all         # params -> train -> visuals -> report

This file parses arguments and calls into :mod:`trafficflow`. It holds no analysis logic,
so everything it invokes stays importable and testable on its own.

Stages talk to each other through files under ``output/``. Detection is the only expensive
step; once it has run, every other stage is seconds of work, so changing a threshold or
re-rendering a figure never means waiting for the detector again.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from trafficflow.config import Paths, load_calibration, load_site
from trafficflow.dataset import TrafficDatabase


def _load(paths: Paths):
    """Load the database and the site facts -- needed by nearly every stage."""
    return TrafficDatabase.load(paths.archive), load_site(paths.config / "site.yaml")


def _read_parameter_rows(paths: Paths) -> list[dict]:
    """Read ``output/parameters/clips.csv``, with a clear message if it is absent."""
    target = paths.parameters / "clips.csv"
    if not target.exists():
        sys.exit(
            f"{target} not found.\n"
            f"Run:  python run.py detect   (once, slow)\n"
            f"then: python run.py params"
        )
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _print_free_flow_check(clips_csv: Path, limit_kmh: float) -> None:
    """Free-flow speed from the fundamental diagram, next to the posted limit."""
    import numpy as np

    from trafficflow.charts import fit_fundamental_diagram

    with clips_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    density = np.array([float(row["density_pc_mi_ln"] or "nan") for row in rows])
    speed = np.array([float(row["speed_median_kmh"] or "nan") for row in rows])
    try:
        fit = fit_fundamental_diagram(density, speed)
    except ValueError as error:
        # Print what actually went wrong: "not enough clips" and "speed does not fall
        # with density" are very different problems and used to read identically.
        print(f"No fundamental diagram: {error}")
        return
    print(
        f"Free-flow speed {fit.free_flow_speed_kmh:.1f} km/h vs posted {limit_kmh:.1f} km/h "
        f"({fit.free_flow_speed_kmh / limit_kmh - 1:+.0%}) -- the external check on the calibration."
    )


# -- stages --------------------------------------------------------------------


def stage_detect(args, paths: Paths) -> None:
    """Detect and track vehicles in every clip. Resumable; skips clips already done."""
    from trafficflow.detector import DetectorConfig, compare_input_sizes, detect_all

    db, site = _load(paths)
    config = DetectorConfig(imgsz=args.imgsz, confidence=args.conf)

    if args.compare_sizes:
        print("Measuring what each model and input size finds, against the configured")
        print(f"{config.weights} at {config.imgsz}. 'kept' is the share of its vehicles a setting")
        print("still finds; 'smallest' is the shortest box, i.e. how far down the road it sees.\n")
        # Two clips per class, spread through the recordings, so heavy traffic and both
        # recording days are in the sample.
        sample = []
        for traffic_class in ("light", "medium", "heavy"):
            members = db.by_class(traffic_class)
            sample += [members[len(members) // 4], members[3 * len(members) // 4]]
        results = compare_input_sizes(sample, db.data_root, site.image, sizes=args.sizes,
                                      config=config, weights=args.weights)
        print(f"  {'weights':<12} {'size':>5} {'on road':>8} {'cars':>6} {'bus/truck':>9} "
              f"{'kept':>6} {'smallest':>9} {'s/frame':>8}")
        for (weights, size), m in results.items():
            print(f"  {weights:<12} {size:>5} {m['on_roadway']:>8.0f} {m['cars']:>6} {m['bus_truck']:>9} "
                  f"{m['kept']:>6.0%} {m['smallest_px']:>7.1f}px {m['seconds_per_frame']:>8.2f}")
        print("\nPick a setting, then rerun without --compare-sizes.")
        return

    started = last = time.perf_counter()

    def progress(done: int, total: int, name: str, rows: int) -> None:
        # One line per clip, flushed, so a log file shows exactly which video it is on.
        nonlocal last
        now = time.perf_counter()
        note = "cached" if rows < 0 else f"{rows} rows"
        eta = (now - started) / done * (total - done)
        print(f"  [{time.strftime('%H:%M:%S')}] [{done}/{total}] {name}  {note}  "
              f"{now - last:.1f}s  ~{eta / 60:.0f} min left", flush=True)
        last = now

    clips = args.clips
    if args.sample and clips is None:
        # Same clips every run, so rerunning the same command resumes rather than starting
        # a different subset. Evenly spaced within each class to span the recording days.
        clips = []
        for i, traffic_class in enumerate(("light", "medium", "heavy")):
            members = db.by_class(traffic_class)
            want = min(len(members), args.sample // 3 + (1 if i < args.sample % 3 else 0))
            clips += [members[k * len(members) // want].name for k in range(want)]
        print(f"Sample of {len(clips)} clips, balanced across light / medium / heavy.")

    print(f"Detecting with {config.weights}, imgsz={config.imgsz}, conf={config.confidence}.")
    written = detect_all(
        db, site.image, config, paths,
        clips=clips, skip_existing=not args.force, progress=progress,
    )
    fresh = sum(1 for value in written.values() if value >= 0)
    print(f"\n{fresh} clips detected, {len(written) - fresh} already present.")
    print(f"Tables in {paths.tracks}")


def stage_params(args, paths: Paths) -> None:
    """Compute the five traffic parameters from the track tables."""
    from trafficflow.pipeline import analyse_all, write_parameters

    db, site = _load(paths)
    try:
        calibration = load_calibration(paths.calibration)
    except FileNotFoundError as error:
        sys.exit(f"{error}")

    def progress(done: int, total: int, name: str) -> None:
        print(f"\r  [{done}/{total}] {name}   ", end="")

    results = analyse_all(db, site, calibration, paths, progress=progress)
    if not results:
        sys.exit("\nNo track tables found. Run `python run.py detect` first.")

    target = write_parameters(results, paths.parameters / "clips.csv")
    print(f"\n\n{len(results)} clips analysed -> {target}")
    _print_free_flow_check(target, site.roadway.speed_limit_kmh)


def stage_train(args, paths: Paths) -> None:
    """Fit the parameter classifier and score it against the baselines."""
    from trafficflow.classify import make_fit_predict
    from trafficflow.evaluate import (
        cross_validate,
        majority_baseline,
        nearest_recording_baseline,
    )

    db, _ = _load(paths)
    rows = {row["clip"]: row for row in _read_parameter_rows(paths)}
    print(f"{len(rows)} clips with parameters.\n")

    # Every score here uses folds that keep each recording run whole.
    results = [majority_baseline(db), nearest_recording_baseline(db)]
    for result in results:
        print(result.summary())

    if len(rows) < len(db):
        print(
            f"NOTE: only {len(rows)} of {len(db)} clips have parameters, so the "
            f"parameter classifier is not directly comparable to the baselines above.\n"
        )

    parameter_model = cross_validate(
        db, make_fit_predict(rows), name="parameter classifier"
    )
    print(parameter_model.summary())

    target = paths.reports / "eval_parameters.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [result.to_dict() for result in [*results, parameter_model]],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {target}")


def stage_visuals(args, paths: Paths) -> None:
    """Render the figures, and annotated video for a few representative clips."""
    from trafficflow import charts

    db, site = _load(paths)
    rows = _read_parameter_rows(paths)

    made = [
        charts.plot_fundamental_diagram(rows, paths.figures / "fundamental_diagram.png"),
        charts.plot_hour_profile(rows, paths.figures / "hour_profile.png"),
        charts.plot_lane_usage(rows, site.roadway.lane_count, paths.figures / "lane_usage.png"),
        charts.plot_speed_distribution(
            rows, site.roadway.speed_limit_kmh, paths.figures / "speed_distribution.png"
        ),
    ]
    for path in made:
        print(f"  {path}")

    if args.videos:
        from trafficflow.pipeline import analyse_clip, road_frame_from
        from trafficflow.overlay import render_overlay
        from trafficflow.tracks import TrackTable

        calibration = load_calibration(paths.calibration)
        print()
        for traffic_class in ("light", "medium", "heavy"):
            clip = next(
                (c for c in db.by_class(traffic_class)
                 if (paths.tracks / f"{c.name}.csv").exists()), None
            )
            if clip is None:
                continue
            table = TrackTable.load(paths.tracks / f"{clip.name}.csv")
            frame = road_frame_from(site, calibration, clip.day_index)
            parameters = analyse_clip(clip, table, frame, site.level_of_service, site.image.fps)
            path = render_overlay(
                clip, table, parameters, db.data_root,
                paths.videos / f"{traffic_class}_{clip.name}.mp4",
            )
            print(f"  {path}")


#: The demo clips use the more accurate checkpoint; the batch pass and live page use the default.
DEMO_WEIGHTS = "yolo26m.pt"


def stage_demo(args, paths: Paths) -> None:
    """The demo clips with ``yolo26m``: annotated videos and a before/after image in demo/.

    Only three clips, so the more accurate checkpoint at a large input size (``--imgsz``,
    1280 by default: distant vehicles stay several pixels across) costs minutes here, where
    a whole pass like that costs many hours. Their tables go to ``output/demo_tracks/``,
    apart from the batch tables, so the dataset-wide results never mix two detectors.
    """
    import cv2
    import numpy as np

    from trafficflow.detector import DetectorConfig, detect_clip, load_model
    from trafficflow.overlay import annotate_frame, clip_badge, render_overlay
    from trafficflow.pipeline import analyse_clip, road_frame_from
    from trafficflow.tracks import TrackTable, stale_fields, write_settings, write_tracks
    from trafficflow.video import read_frame

    db, site = _load(paths)
    calibration = load_calibration(paths.calibration)
    config = DetectorConfig(weights=DEMO_WEIGHTS, imgsz=args.imgsz)
    demo, tracks = paths.root / "demo", paths.output / "demo_tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    print(f"Demo clips with {config.weights}, imgsz={config.imgsz}, conf={config.confidence}.")

    model, panels = None, []
    for traffic_class in ("light", "medium", "heavy"):
        clip = db.by_class(traffic_class)[0]
        table_path = tracks / f"{clip.name}.csv"
        if not table_path.exists() or stale_fields(table_path, config):
            if model is None:
                model = load_model(config)
            started = time.perf_counter()
            write_tracks(table_path, detect_clip(clip, db.data_root, site.image, config, model=model))
            write_settings(table_path, config)
            print(f"  detected {clip.name} in {time.perf_counter() - started:.0f}s", flush=True)

        table = TrackTable.load(table_path)
        frame = road_frame_from(site, calibration, clip.day_index)
        parameters = analyse_clip(clip, table, frame, site.level_of_service, site.image.fps)
        print(f"  {render_overlay(clip, table, parameters, db.data_root, demo / f'{traffic_class}_{clip.name}.mp4')}")

        if traffic_class in ("light", "heavy"):
            # The busiest frame of the clip, raw beside annotated.
            number, rows = max(table.iter_frames(), key=lambda item: len(item[1]))
            image = read_frame(clip.video_path(db.data_root), int(number))
            speeds = {estimate.track_id: estimate for estimate in parameters.speeds}
            raw = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            panels.append(np.hstack([raw, annotate_frame(image, rows, speeds, clip_badge(parameters))]))

    target = demo / "what_the_shapes_are.png"
    cv2.imwrite(str(target), np.vstack(panels))
    print(f"  {target}")

    for written in _export_notebook_results(paths.root / "notebooks", demo / "notebooks"):
        print(f"  {written}")


def _export_notebook_results(notebooks: Path, target: Path) -> list[Path]:
    """Copy the saved Colab outputs of every notebook into ``target``.

    Each figure is named after the notebook and the section heading above it, and the text
    results go to one ``results.md``, so a reader sees what the GPU stages produced without
    opening Jupyter. Install logs, warnings and progress bars are left out.
    """
    import base64
    import re

    noise = re.compile(r"warn|Download|%\||\x1b|\[\?25|torch\.onnx|return cls|Building wheel|<Figure|<IPython"
                       r"|NystromAttention|pretrained weights|EdgeGuidedLocalSSI|^\s*$", re.I)
    target.mkdir(parents=True, exist_ok=True)
    written, lines = [], ["# Notebook results", "", "Saved outputs of the Colab runs, copied by `python run.py demo`.", ""]
    for notebook in sorted(notebooks.glob("*.ipynb")):
        lines += [f"## {notebook.name}", ""]
        heading, shown = notebook.stem, None
        for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
            source = "".join(cell["source"]).strip()
            if cell["cell_type"] == "markdown" and source.startswith("#"):
                heading = source.splitlines()[0].lstrip("#").strip()
                continue
            images, texts = [], []
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" in data:
                    slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
                    path = target / f"{notebook.stem}_{slug}.png"
                    n = 2
                    while path in written:  # a second figure under the same heading
                        path, n = target / f"{notebook.stem}_{slug}_{n}.png", n + 1
                    path.write_bytes(base64.b64decode(data["image/png"]))
                    written.append(path)
                    images.append(path)
                text = "".join(output.get("text", data.get("text/plain", "")))
                texts += [line for line in text.splitlines() if not noise.search(line)]
            if not (images or texts):
                continue
            if heading != shown:
                lines += [f"### {heading}", ""]
                shown = heading
            lines += [line for path in images for line in (f"![{heading}]({path.name})", "")]
            if texts:
                lines += ["```", *texts, "```", ""]
    report = target / "results.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return [*written, report]


def stage_report(args, paths: Paths) -> None:
    """Write the insights and recommendations."""
    from trafficflow.insights import build_report, write_report

    _, site = _load(paths)
    calibration = load_calibration(paths.calibration)
    rows = _read_parameter_rows(paths)

    scores = None
    for candidate in (paths.models / "eval_finetune.json", paths.reports / "eval_finetune.json"):
        if candidate.exists():
            scores = json.loads(candidate.read_text(encoding="utf-8"))
            break

    target = write_report(
        build_report(rows, site, calibration, scores), paths.reports / "insights.md"
    )
    print(f"  {target}")


def stage_serve(args, paths: Paths) -> None:
    """The website: live analysis of a clip or an upload, and the whole-dataset results."""
    import threading
    import webbrowser

    import uvicorn

    from app.server import create_app

    url = f"http://{args.host}:{args.port}"
    print(f"Loading the models, then the website opens at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(paths, warm=True), host=args.host, port=args.port, log_level="warning")


def stage_all(args, paths: Paths) -> None:
    """Everything after detection, in order."""
    for name, stage in (
        ("params", stage_params), ("train", stage_train),
        ("visuals", stage_visuals), ("report", stage_report),
    ):
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        stage(args, paths)


# -- command line --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `detect` once; everything after it reads from output/ in seconds.",
    )
    parser.add_argument("--root", type=Path, default=None, help="project root (default: this file's directory)")
    sub = parser.add_subparsers(dest="stage", required=True)

    detect = sub.add_parser("detect", help="YOLO + ByteTrack over every clip (slow, once)")
    # Imported here rather than at the top: the defaults the help text quotes belong to
    # the detector, and there must not be a second copy of them in the CLI.
    from trafficflow.detector import DetectorConfig

    defaults = DetectorConfig()
    detect.add_argument("--imgsz", type=int, default=defaults.imgsz,
                        help="inference size; the clips are 320x240 but distant vehicles "
                             f"vanish at that size (default: {defaults.imgsz})")
    detect.add_argument("--conf", type=float, default=defaults.confidence,
                        help=f"detection confidence (default: {defaults.confidence})")
    detect.add_argument("--clips", nargs="*", default=None, help="limit to these clip names")
    detect.add_argument("--sample", type=int, default=None,
                        help="detect a fixed class-balanced sample of N clips (resumable)")
    detect.add_argument("--force", action="store_true", help="re-detect clips that already have a table")
    detect.add_argument("--compare-sizes", action="store_true",
                        help="measure detections per model and input size, then exit")
    detect.add_argument("--sizes", type=int, nargs="+", default=[640, 960, 1280],
                        help="input sizes for --compare-sizes (default: 640 960 1280)")
    detect.add_argument("--weights", nargs="+", default=[defaults.weights],
                        help="checkpoints for --compare-sizes, e.g. yolo11n.pt yolo26m.pt "
                             f"(default: {defaults.weights})")
    detect.set_defaults(func=stage_detect)

    params = sub.add_parser("params", help="compute the five traffic parameters")
    params.set_defaults(func=stage_params)

    train = sub.add_parser("train", help="fit and score the parameter classifier")
    train.set_defaults(func=stage_train)

    visuals = sub.add_parser("visuals", help="render figures and annotated video")
    visuals.add_argument("--videos", action="store_true", help="also render annotated clips")
    visuals.set_defaults(func=stage_visuals)

    demo = sub.add_parser("demo", help="demo clips with yolo26m: videos and image in demo/")
    demo.add_argument("--imgsz", type=int, default=1280,
                      help="inference size for the demo clips (default: 1280, for quality)")
    demo.set_defaults(func=stage_demo)

    report = sub.add_parser("report", help="write insights and recommendations")
    report.set_defaults(func=stage_report)

    serve = sub.add_parser("serve", help="the website: live analysis and results")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    serve.set_defaults(func=stage_serve)

    everything = sub.add_parser("all", help="params, train, visuals, report")
    everything.add_argument("--videos", action="store_true")
    everything.set_defaults(func=stage_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = (Paths(args.root) if args.root else Paths()).ensure()
    args.func(args, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
