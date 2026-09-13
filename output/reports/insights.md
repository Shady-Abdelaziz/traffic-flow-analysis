# Traffic Flow Analysis — I-5 southbound at S 188th St

*Generated 2026-09-13 from 254 clips.*

**Site.** Interstate 5 southbound at S 188th St, Seattle, Washington. 5 lanes across 18.15 m of travelled way, measured from the footage rather than taken from a road inventory. Posted limit 96.6 km/h. Measurements are confined to the 59-156 m measurement zone, which runs from the bottom of the frame to the far end of the traced markings.

**Clips analysed.** 165 light, 45 medium, 44 heavy.

## 1. Fundamental diagram — the headline finding

Every clip is one point: density against median speed. The relationship between them, not either number alone, is what characterises a road.

| Quantity | Measured |
|---|---|
| Free-flow speed | **98.4 km/h** |
| Critical density | **30.9 pc/mi/ln** |
| Capacity | **944 veh/h/ln** |
| Jam density | 61.8 pc/mi/ln |
| Fit quality (R²) | 0.757 over 223 clips |

**The critical density is the number that matters.** Below 31 pc/mi/ln, adding vehicles adds throughput. Above it, adding vehicles *reduces* throughput — the segment is past the point where more traffic means more flow.

The fitted free-flow speed of 98.4 km/h can be checked against something external: the posted limit is 96.6 km/h. The two agree to within 2%, which is independent evidence that the scale is right.

This figure also validates the calibration. The shape has been known since Greenshields in 1935; had the pixel-to-metre mapping been wrong, the points would scatter rather than trace it. With no ground-truth speeds in the database, that is the strongest check available.

## 2. When this road fails

| Hour | Clips | Speed (km/h) | Density (pc/mi/ln) | Worst LOS |
|---|---|---|---|---|
| 06:00 | 14 | 92 | 5.5 | A |
| 07:00 | 14 | 92 | 2.1 | A |
| 08:00 | 10 | 93 | 4.6 | A |
| 09:00 | 17 | 85 | 3.2 | A |
| 10:00 | 15 | 85 | 4.6 | B |
| 11:00 | 14 | 85 | 4.1 | B |
| 12:00 | 15 | 81 | 4.7 | B |
| 13:00 | 11 | 89 | 14.4 | D |
| 14:00 | 9 | 72 | 17.6 | E |
| 15:00 | 16 | 35 | 29.6 | F |
| 16:00 | 30 | 33 | 38.0 | F |
| 17:00 | 28 | 24 | 43.4 | F |
| 18:00 | 20 | 59 | 26.4 | E |
| 19:00 | 31 | 93 | 10.6 | C |
| 20:00 | 10 | 93 | 10.5 | B |

**Peak congestion is at 17:00**, at 43.4 pc/mi/ln. The hour is recoverable from each filename, so the whole database can be placed on a daily profile without any external timetable.

## 3. Lane behaviour

Lane 1 is the HOV lane. Whether it keeps moving while the general-purpose lanes saturate is the most directly actionable question in this dataset.

| Lane | Occupancy, light | Occupancy, heavy |
|---|---|---|
| 1 (HOV) | 0.100 | 0.891 |
| 2 | 0.236 | 0.953 |
| 3 | 0.420 | 0.941 |
| 4 | 0.425 | 0.912 |
| 5 | 0.270 | 0.859 |

Mean lane imbalance is 0.411. A value near zero means traffic spreads evenly; a large one means it does not, which on a freeway usually points at a downstream bottleneck or at drivers avoiding a lane.

## 4. Vehicle mix and lane-changing

Buses and trucks are **12.2%** of vehicles. They are counted as two passenger-car equivalents in the density figures, following HCM practice for level terrain — ignoring that would understate congestion on a corridor carrying freight.

| Traffic class | Lane changes per vehicle |
|---|---|
| light | 0.001 |
| medium | 0.010 |
| heavy | 0.006 |

Lane-changing is expected to peak at moderate density and fall away at high density, where there is no longer a gap to move into. Whether the measurements show that is visible above.

## 5. Weather — and why the obvious reading is wrong

| Weather | Clips | % light | Mean speed (km/h) |
|---|---|---|---|
| clear | 51 | 47% | 59 |
| overcast | 125 | 54% | 62 |
| rain | 78 | 95% | 84 |

**This table must not be read as a weather effect.** In this database 74 of the 78 rain clips are labelled light — not because rain clears traffic, but because it rained outside the peak. Weather is confounded with time of day, so any comparison has to be made within a single hour, and on 254 clips there is not enough data in any one hour to do that convincingly.

## 6. Congestion classification

| Model | Accuracy | Macro-F1 |
|---|---|---|
| majority class | 65.0% | 0.263 |
| nearest recording (no pixels) | 71.3% | 0.584 |
| fine-tuned resnet18 | 93.7% | 0.888 |

**The bar is a model that reads no pixels at all.** Each clip is a five-second recording and the next clip is the next five seconds, so scores here use the grouped by recording split, which keeps whole recordings in one fold. Even then, copying the nearest recording's label scores 0.584 macro-F1. Macro-F1 is reported alongside accuracy because the classes are 165/45/44 — the majority baseline reaches 65% accuracy while scoring 0.000 on 2 of the 3 classes.

## 7. Recommendations

**1. Meter the S 188th St on-ramp during 13:00–19:00.** 77 of 254 clips exceed the measured critical density of 31 pc/mi/ln. Metering holds arriving traffic just below that point, which keeps throughput near the measured capacity of 944 veh/h/ln instead of letting it collapse.

**2. Use the measured critical density as the control threshold, not a default.** The HCM's generic capacity is 2300 veh/h/ln; this segment measures 944. Tuning control to a figure measured on this road is better than tuning to a national average.

**3. Check HOV lane use before adding capacity.** If lane 1 is much emptier than the general-purpose lanes at the peak, the corridor's problem is distribution rather than lack of lanes, and occupancy rules or enforcement cost far less than construction.

## 8. What this data cannot support

**Clips are 5.3 seconds and not contiguous.** Every density figure is an instantaneous snapshot, whereas the HCM thresholds were calibrated on 15-minute averages. The Level of Service grades are meaningful, but more scattered than a 15-minute figure would be. Queue length, travel time, and how congestion evolves over minutes are simply not measurable here.

**Turning.** Not applicable. This is an Interstate mainline: traffic is one-way and there are no turning movements to measure. Vehicles leave only at ramps.

**Merging.** Not measurable from this viewpoint. No junction enters the frame -- the edge markings run unbroken to the far end of the visible road -- so any merge happens beyond the measurement zone, at a range where vehicles are only a few pixels across. Merge detection is implemented and run; a near-zero count is the expected and honest result.

**Motorcycles.** At 320×240 a motorcycle is a few pixels across and is not reliably detected. Counts for that class should be read as a lower bound.

**Speeds have no ground truth.** No measured speeds ship with this database, so accuracy is argued from internal consistency — agreement between independent scale estimates, the fitted free-flow speed against the posted limit, the expected ordering light > medium > heavy, and the shape of the fundamental diagram — rather than demonstrated against a reference.