"""Turning measured parameters into written findings and recommendations.

This module produces the analysis a traffic engineer would actually read: what the
numbers say about this stretch of road, what should be done about it, and -- equally
important -- what this data cannot support.

The organising finding is the **fundamental diagram**. It yields two quantities that
exist in no file anywhere and come only from our own measurements: the density at which
this segment breaks down, and its real capacity. Every other finding is positioned
relative to those.

Limits are stated rather than omitted. A report that quietly drops the parameters it
could not measure is less trustworthy than one that says which they were and why.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from trafficflow.charts import fit_fundamental_diagram
from trafficflow.config import Calibration, SiteConfig
from trafficflow.pipeline import zone_bounds_m
from trafficflow.trajectory import MERGING_NOTE, TURNING_NOTE

__all__ = [
    "build_report", "write_report", "live_line", "recommendation_list",
    "classification_summary", "build_results",
    "FREE_FLOWING_RATIO", "SLOWING_RATIO",
]


#: Speed as a fraction of the posted limit at or above which traffic reads as free
#: flowing, and the fraction below which it reads as stop-and-go. Anything between the
#: two is "slowing". These are the one place the live sentence's three states are
#: defined -- any other view of the same states should import them rather than pick
#: its own cut points, because two views disagreeing about what "slowing" means is a
#: bug the reader sees and cannot explain.
FREE_FLOWING_RATIO = 0.8
SLOWING_RATIO = 0.5


def _f(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(rows: list[dict], column: str) -> float:
    values = np.array([_f(row.get(column)) for row in rows])
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def build_report(
    rows: list[dict],
    site: SiteConfig,
    calibration: Calibration,
    model_scores: dict | None = None,
) -> str:
    """Compose the full analysis as Markdown.

    Parameters
    ----------
    rows:
        Parameter rows, one per clip, as written by :mod:`trafficflow.pipeline`.
    site:
        The lane count, and the posted limit to compare measurements against.
    calibration:
        The road geometry, measured from the footage.
    model_scores:
        Optional contents of ``eval_finetune.json``, so the model results appear in
        the same report as the measurements they are compared against.
    """
    density = np.array([_f(row["density_pc_mi_ln"]) for row in rows])
    speed = np.array([_f(row["speed_median_kmh"]) for row in rows])

    parts = [
        _header(rows, site, calibration),
        _fundamental(density, speed, site),
        _daily(rows),
        _lanes(rows, site),
        _mix_and_behaviour(rows),
        _weather(rows),
        _models(model_scores),
        _recommendations(rows, density, speed, site),
        _limits(rows),
    ]
    return "\n\n".join(part for part in parts if part)


def _header(rows: list[dict], site: SiteConfig, calibration: Calibration) -> str:
    counts = Counter(row["traffic_class"] for row in rows)
    near_m, far_m = zone_bounds_m(site, calibration)
    return "\n".join(
        [
            "# Traffic Flow Analysis — I-5 southbound at S 188th St",
            "",
            f"*Generated {date.today().isoformat()} from {len(rows)} clips.*",
            "",
            f"**Site.** {site.location['description']}. "
            f"{site.roadway.lane_count} lanes across "
            f"{calibration.measured_width_m:.2f} m of travelled way, measured from the "
            f"footage rather than taken from a road inventory. "
            f"Posted limit {site.roadway.speed_limit_kmh:.1f} km/h. "
            f"Measurements are confined to the "
            f"{near_m:.0f}-{far_m:.0f} m measurement zone, which runs from the bottom "
            f"of the frame to the far end of the traced markings.",
            "",
            f"**Clips analysed.** {counts.get('light', 0)} light, "
            f"{counts.get('medium', 0)} medium, {counts.get('heavy', 0)} heavy.",
        ]
    )


def _fundamental(density: np.ndarray, speed: np.ndarray, site: SiteConfig) -> str:
    try:
        fit = fit_fundamental_diagram(density, speed)
    except ValueError as error:
        return (
            "## 1. Fundamental diagram\n\n"
            f"No fit: {error}. Re-run the detection stage before relying on this section."
        )

    return "\n".join(
        [
            "## 1. Fundamental diagram — the headline finding",
            "",
            "Every clip is one point: density against median speed. The relationship "
            "between them, not either number alone, is what characterises a road.",
            "",
            "| Quantity | Measured |",
            "|---|---|",
            f"| Free-flow speed | **{fit.free_flow_speed_kmh:.1f} km/h** |",
            f"| Critical density | **{fit.critical_density:.1f} pc/mi/ln** |",
            f"| Capacity | **{fit.capacity_veh_per_h_per_ln:.0f} veh/h/ln** |",
            f"| Jam density | {fit.jam_density:.1f} pc/mi/ln |",
            f"| Fit quality (R²) | {fit.r_squared:.3f} over {fit.n_points} clips |",
            "",
            f"**The critical density is the number that matters.** Below "
            f"{fit.critical_density:.0f} pc/mi/ln, adding vehicles adds throughput. Above it, "
            "adding vehicles *reduces* throughput — the segment is past the point where more "
            "traffic means more flow.",
            "",
            f"The fitted free-flow speed of {fit.free_flow_speed_kmh:.1f} km/h can be checked "
            f"against something external: the posted limit is {site.roadway.speed_limit_kmh:.1f} km/h. "
            + _agreement_note(fit.free_flow_speed_kmh, site.roadway.speed_limit_kmh),
            "",
            "This figure also validates the calibration. The shape has been known since "
            "Greenshields in 1935; had the pixel-to-metre mapping been wrong, the points "
            "would scatter rather than trace it. With no ground-truth speeds in the "
            "database, that is the strongest check available.",
        ]
    )


def _agreement_note(measured: float, limit: float) -> str:
    error = abs(measured - limit) / limit * 100
    if error < 10:
        return (
            f"The two agree to within {error:.0f}%, which is independent evidence that the "
            "scale is right."
        )
    return (
        f"They differ by {error:.0f}%. That gap is a calibration warning, not a finding about "
        "driver behaviour, and the scale estimates should be revisited before these speeds "
        "are quoted."
    )


def _daily(rows: list[dict]) -> str:
    hours = sorted({int(row["hour"]) for row in rows})
    lines = [
        "## 2. When this road fails",
        "",
        "| Hour | Clips | Speed (km/h) | Density (pc/mi/ln) | Worst LOS |",
        "|---|---|---|---|---|",
    ]
    worst_hour, worst_density = None, -1.0
    for hour in hours:
        subset = [row for row in rows if int(row["hour"]) == hour]
        mean_speed, mean_density = _mean(subset, "speed_median_kmh"), _mean(subset, "density_pc_mi_ln")
        worst_los = max((row["level_of_service"] for row in subset), default="-")
        lines.append(
            f"| {hour:02d}:00 | {len(subset)} | "
            f"{'—' if mean_speed != mean_speed else f'{mean_speed:.0f}'} | "
            f"{'—' if mean_density != mean_density else f'{mean_density:.1f}'} | {worst_los} |"
        )
        if mean_density == mean_density and mean_density > worst_density:
            worst_density, worst_hour = mean_density, hour

    if worst_hour is not None:
        lines += [
            "",
            f"**Peak congestion is at {worst_hour:02d}:00**, at {worst_density:.1f} pc/mi/ln. "
            "The hour is recoverable from each filename, so the whole database can be placed "
            "on a daily profile without any external timetable.",
        ]
    return "\n".join(lines)


def _lanes(rows: list[dict], site: SiteConfig) -> str:
    lanes = range(1, site.roadway.lane_count + 1)
    lines = [
        "## 3. Lane behaviour",
        "",
        "Lane 1 is the HOV lane. Whether it keeps moving while the general-purpose lanes "
        "saturate is the most directly actionable question in this dataset.",
        "",
        "| Lane | Occupancy, light | Occupancy, heavy |",
        "|---|---|---|",
    ]
    light = [row for row in rows if row["traffic_class"] == "light"]
    heavy = [row for row in rows if row["traffic_class"] == "heavy"]
    for lane in lanes:
        column = f"occupancy_lane{lane}"
        name = f"{lane} (HOV)" if lane == 1 else str(lane)
        lines.append(f"| {name} | {_mean(light, column):.3f} | {_mean(heavy, column):.3f} |")

    imbalance = _mean(rows, "lane_imbalance")
    lines += [
        "",
        f"Mean lane imbalance is {imbalance:.3f}. A value near zero means traffic spreads "
        "evenly; a large one means it does not, which on a freeway usually points at a "
        "downstream bottleneck or at drivers avoiding a lane.",
    ]
    return "\n".join(lines)


def _mix_and_behaviour(rows: list[dict]) -> str:
    heavy_share = _mean(rows, "heavy_vehicle_percent")
    lines = [
        "## 4. Vehicle mix and lane-changing",
        "",
        f"Buses and trucks are **{heavy_share:.1f}%** of vehicles. They are counted as two "
        "passenger-car equivalents in the density figures, following HCM practice for level "
        "terrain — ignoring that would understate congestion on a corridor carrying freight.",
        "",
        "| Traffic class | Lane changes per vehicle |",
        "|---|---|",
    ]
    for name in ("light", "medium", "heavy"):
        subset = [row for row in rows if row["traffic_class"] == name]
        if subset:
            lines.append(f"| {name} | {_mean(subset, 'changes_per_vehicle'):.3f} |")
    lines += [
        "",
        "Lane-changing is expected to peak at moderate density and fall away at high "
        "density, where there is no longer a gap to move into. Whether the measurements "
        "show that is visible above.",
    ]
    return "\n".join(lines)


def _weather(rows: list[dict]) -> str:
    lines = [
        "## 5. Weather — and why the obvious reading is wrong",
        "",
        "| Weather | Clips | % light | Mean speed (km/h) |",
        "|---|---|---|---|",
    ]
    for weather in sorted({row["weather"] for row in rows}):
        subset = [row for row in rows if row["weather"] == weather]
        share = 100 * sum(1 for row in subset if row["traffic_class"] == "light") / len(subset)
        lines.append(
            f"| {weather} | {len(subset)} | {share:.0f}% | {_mean(subset, 'speed_median_kmh'):.0f} |"
        )
    # Counted rather than quoted: the claim is about these rows, so it is measured from
    # them and stays true if the set of clips ever changes.
    rain = [row for row in rows if "rain" in row["weather"].lower()]
    rain_light = sum(1 for row in rain if row["traffic_class"] == "light")
    lines += [
        "",
        f"**This table must not be read as a weather effect.** In this database "
        f"{rain_light} of the {len(rain)} rain clips are labelled light — not because "
        "rain clears traffic, but because it rained outside the peak. Weather is "
        "confounded with time of day, so any comparison has to be made within a single "
        f"hour, and on {len(rows)} clips there is not enough data in any one hour to do "
        "that convincingly.",
    ]
    return "\n".join(lines)


def _score(scores: dict, split: str, name: str, field: str) -> float:
    """One number off the scoreboard, or NaN when that row was not scored."""
    for result in scores.get("results", []):
        if result.get("split") == split and name in result.get("name", ""):
            return float(result[field])
    return float("nan")


def _models(scores: dict | None) -> str:
    if not scores:
        return ""
    lines = [
        "## 6. Congestion classification",
        "",
        "| Model | Accuracy | Macro-F1 |",
        "|---|---|---|",
    ]
    for result in scores.get("results", []):
        lines.append(
            f"| {result['name']} | {result['accuracy']:.1%} | {result['macro_f1']:.3f} |"
        )

    # Every figure in the paragraph below is read back out of the eval file rather than
    # typed in. They were literals until the numbers and the prose could drift apart.
    headline = scores.get("headline", "grouped by recording")
    probe = _score(scores, headline, "nearest recording", "macro_f1")
    majority = _score(scores, headline, "majority", "accuracy")

    # Class counts are the confusion matrix's row sums -- the truth totals.
    confusion = scores.get("confusion", {}).get(headline) or []
    classes = list(scores.get("preprocessing", {}).get("classes", []))
    counts = [sum(row) for row in confusion]
    split_text = "/".join(str(c) for c in counts) if counts else "imbalanced"
    unscored = max(0, len(classes) - 1)

    lines += [
        "",
        "**The bar is a model that reads no pixels at all.** Each clip is a five-second "
        "recording and the next clip is the next five seconds, so scores here use the "
        f"{headline} split, which keeps whole recordings in one fold. Even then, copying "
        f"the nearest recording's label scores {probe:.3f} macro-F1. Macro-F1 is "
        "reported alongside accuracy "
        f"because the classes are {split_text} — the majority baseline reaches "
        f"{majority:.0%} accuracy while scoring 0.000 on {unscored} of the "
        f"{len(classes)} classes.",
    ]
    return "\n".join(lines)


def recommendation_list(rows: list[dict], density: np.ndarray, speed: np.ndarray, site: SiteConfig) -> list[dict]:
    """The three recommendations as plain title + text, for the report and the website."""
    try:
        fit = fit_fundamental_diagram(density, speed)
    except ValueError:
        return []
    critical, capacity = fit.critical_density, fit.capacity_veh_per_h_per_ln
    over = [row for row in rows if _f(row["density_pc_mi_ln"]) > critical]
    # Uploaded videos carry no hour of day.
    hours_over = sorted({int(row["hour"]) for row in over if row["hour"] not in (None, "")})
    window = f"{min(hours_over):02d}:00–{max(hours_over) + 1:02d}:00" if hours_over else "the peak"
    return [
        {"title": f"Meter the S 188th St on-ramp during {window}.",
         "text": f"{len(over)} of {len(rows)} clips exceed the measured critical density of "
                 f"{critical:.0f} pc/mi/ln. Metering holds arriving traffic just below that point, "
                 f"which keeps throughput near the measured capacity of {capacity:.0f} veh/h/ln "
                 "instead of letting it collapse."},
        {"title": "Use the measured critical density as the control threshold, not a default.",
         "text": f"The HCM's generic capacity is {site.level_of_service.capacity_veh_per_h_per_ln:.0f} "
                 f"veh/h/ln; this segment measures {capacity:.0f}. Tuning control to a figure measured "
                 "on this road is better than tuning to a national average."},
        {"title": "Check HOV lane use before adding capacity.",
         "text": "If lane 1 is much emptier than the general-purpose lanes at the peak, the corridor's "
                 "problem is distribution rather than lack of lanes, and occupancy rules or enforcement "
                 "cost far less than construction."},
    ]


def _recommendations(rows: list[dict], density: np.ndarray, speed: np.ndarray, site: SiteConfig) -> str:
    items = recommendation_list(rows, density, speed, site)
    if not items:
        return ""
    body = [f"**{n}. {item['title']}** {item['text']}" for n, item in enumerate(items, start=1)]
    return "## 7. Recommendations\n\n" + "\n\n".join(body)


# -- the website -----------------------------------------------------------------


def live_line(speed_kmh: float | None, level_of_service: str | None,
              congestion: str | None, limit_kmh: float) -> str:
    """One sentence for the live video, from the numbers so far.

    The two thresholds are :data:`FREE_FLOWING_RATIO` and :data:`SLOWING_RATIO`, named
    rather than written inline so that anything else describing the same three states
    can refer to them instead of choosing its own numbers.
    """
    if speed_kmh is None:
        return "Measuring — speeds appear once a vehicle has been tracked for a second."
    ratio = speed_kmh / limit_kmh
    state = ("Free-flowing" if ratio >= FREE_FLOWING_RATIO
             else "Slowing" if ratio >= SLOWING_RATIO
             else "Stop-and-go")
    parts = [f"{state}: {speed_kmh:.0f} km/h, {ratio:.0%} of the limit"]
    if level_of_service:
        parts.append(f"LOS {level_of_service}")
    if congestion:
        parts.append(f"CNN reads {congestion}")
    return " · ".join(parts)


def classification_summary(scores: dict | None) -> dict | None:
    """The congestion CNN's headline score, recall per class and confusion matrix."""
    if not scores:
        return None
    split = scores["headline"]

    def row(name: str) -> dict:
        """The scoreboard line for one model on the headline split.

        A bare ``next`` here raised ``StopIteration`` when a model was renamed --
        inside a request handler that becomes an opaque 500 with nothing in it about
        the eval file. The substring match stays, because the exported names carry
        their architecture ("fine-tuned resnet18"), but a miss now says so.
        """
        found = [r for r in scores["results"] if r["split"] == split and name in r["name"]]
        if not found:
            available = ", ".join(sorted(r["name"] for r in scores["results"]
                                         if r["split"] == split)) or "nothing"
            raise KeyError(
                f"no model matching {name!r} was scored on the {split!r} split "
                f"(found: {available}) -- the eval file and this report disagree"
            )
        return found[0]

    classes = list(scores["preprocessing"]["classes"])
    confusion = scores["confusion"][split]
    return {
        "split": split,
        "accuracy": row("resnet")["accuracy"],
        "macro_f1": row("resnet")["macro_f1"],
        "majority_accuracy": row("majority")["accuracy"],
        "classes": classes,
        "confusion": confusion,
        "recall": {name: confusion[i][i] / sum(confusion[i]) for i, name in enumerate(classes)},
    }


def build_results(rows: list[dict], site: SiteConfig, scores: dict | None = None) -> dict:
    """Everything the Results tab draws, as plain JSON-ready values."""
    density = np.array([_f(row["density_pc_mi_ln"]) for row in rows])
    speed = np.array([_f(row["speed_median_kmh"]) for row in rows])
    try:
        fit = fit_fundamental_diagram(density, speed)
        fundamental = {
            "free_flow_speed_kmh": fit.free_flow_speed_kmh,
            "capacity_veh_per_h_per_ln": fit.capacity_veh_per_h_per_ln,
            "critical_density": fit.critical_density,
            "jam_density": fit.jam_density,
            "r_squared": fit.r_squared,
        }
    except ValueError:
        fundamental = None

    # An uploaded video has no dataset entry, so it has no hand-given class and no hour
    # parsed from a clip name. Those stay None rather than becoming a placeholder that
    # would colour a point or invent a time of day.
    points = [
        {"clip": row["clip"], "traffic_class": row["traffic_class"] or None,
         "hour": int(row["hour"]) if row["hour"] not in (None, "") else None,
         "density": float(d), "speed": float(s)}
        for row, d, s in zip(rows, density, speed) if np.isfinite(d) and np.isfinite(s)
    ]
    hours = sorted({point["hour"] for point in points if point["hour"] is not None})
    by_hour = [
        {"hour": hour,
         "median_speed_kmh": float(np.median([p["speed"] for p in points if p["hour"] == hour])),
         "clips": sum(p["hour"] == hour for p in points)}
        for hour in hours
    ]
    los = Counter(row["level_of_service"] for row in rows)
    try:
        classification = classification_summary(scores)
    except KeyError:
        # A renamed or half-written eval file should cost the page its model card,
        # not the whole Results tab. The card's absence is visible; a 500 is not.
        classification = None
    return {
        "clips": len(rows),
        # Uploads are the rows with no dataset class; the page shows them apart from the 254.
        "uploads": sum(1 for row in rows if not row["traffic_class"]),
        "limit_kmh": site.roadway.speed_limit_kmh,
        "fundamental": fundamental,
        "points": points,
        "by_hour": by_hour,
        "los": {grade: los.get(grade, 0) for grade in "ABCDEF"},
        "classification": classification,
        "recommendations": recommendation_list(rows, density, speed, site),
    }


def _limits(rows: list[dict]) -> str:
    return "\n".join(
        [
            "## 8. What this data cannot support",
            "",
            "**Clips are 5.3 seconds and not contiguous.** Every density figure is an "
            "instantaneous snapshot, whereas the HCM thresholds were calibrated on 15-minute "
            "averages. The Level of Service grades are meaningful, but more scattered than a "
            "15-minute figure would be. Queue length, travel time, and how congestion evolves "
            "over minutes are simply not measurable here.",
            "",
            f"**Turning.** {TURNING_NOTE}",
            "",
            f"**Merging.** {MERGING_NOTE}",
            "",
            "**Motorcycles.** At 320×240 a motorcycle is a few pixels across and is not "
            "reliably detected. Counts for that class should be read as a lower bound.",
            "",
            "**Speeds have no ground truth.** No measured speeds ship with this database, so "
            "accuracy is argued from internal consistency — agreement between independent "
            "scale estimates, the fitted free-flow speed against the posted limit, the "
            "expected ordering light > medium > heavy, and the shape of the fundamental "
            "diagram — rather than demonstrated against a reference.",
        ]
    )


def write_report(text: str, path: Path | str) -> Path:
    """Write the report to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
