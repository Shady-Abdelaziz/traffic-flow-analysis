# Demo

| File | What it shows |
|---|---|
| `light_cctv052x2004080516x01640.mp4` | annotated light-traffic clip: boxes, ids, types, speeds · `yolo26m` |
| `medium_cctv052x2004080516x01638.mp4` | the same for medium traffic · `yolo26m` |
| `heavy_cctv052x2004080516x01646.mp4` | the same for heavy traffic · `yolo26m` |
| `what_the_shapes_are.png` | the busiest frame of the light and heavy clips, raw beside annotated · `yolo26m` |
| `fundamental_diagram.png` | speed against density over all 254 clips, with the fitted curve · batch pass, `yolo11n` |
| `hour_profile.png` | level of service by hour of day · batch pass, `yolo11n` |
| `lane_usage.png` | lane occupancy · batch pass, `yolo11n` |
| `speed_distribution.png` | speeds per class against the posted limit · batch pass, `yolo11n` |
| `notebooks/` | the Colab notebooks' saved figures and `results.md` with their text results (calibration, fine-tune scores) |

The three videos and the before/after image come from `python run.py demo`, which detects just
those clips with the default checkpoint (`yolo26m`) at imgsz 1280, so distant vehicles are
still found. On three clips that takes minutes; over all 254 it would take many hours. The four charts cover all
254 clips, so they come from the batch pass (`run.py detect` then `run.py all`).
| `website_demo.mp4` | *to record:* the website live on a heavy clip, an upload, and the Results tab |
| `live_heavy.png` · `live_light.png` · `results.png` | *to capture:* screenshots of the website |

The annotated clips are written as mp4v, which Windows' player and VLC open; browsers may not.

## Recording the website (≈ 90 s)

1. `python run.py serve` and wait for the page.
2. Win + Alt + R to start recording (Xbox Game Bar).
3. Live tab → Heavy → first clip. Let it finish, press Replay.
4. Drop any 5–6 s video onto the drop zone.
5. Results tab, scroll to the classifier card and the recommendations.
6. Win + Alt + R to stop; move the file here as `website_demo.mp4`.
7. Win + Shift + S for the three screenshots.

## Interview script (≈ 5 min)

1. **Problem.** Measure traffic from one fixed 320×240 camera with no ground truth.
2. **Metres from one camera.** Painted edges → horizon; 18.15 m road width → camera height;
   UniDepth → focal length. Check that used no speeds: free-flow 98.4 km/h vs 96.6 posted (+2%).
3. **Live demo.** YOLO + ByteTrack every frame; every second `pipeline.measure` — the same
   function the batch uses — gives speed (RANSAC fit), vehicles observed (≥ 5 frames in zone),
   density → LOS; the CNN classifies 3 frames 1 s apart.
4. **Results.** Critical density 31 pc/mi/ln, level of service F from 15:00 to 18:00 →
   ramp metering.
5. **Accuracy.** 93.7% / 0.888 macro-F1 with whole recordings held out, against a 65% floor and a
   0.584 no-pixels bar.
6. **Scope.** 5 s clips measure the moment recorded; turning does not occur on this mainline.

## Submission zip (without the dataset)

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
$items = Get-ChildItem -Force | Where-Object { $_.Name -notin @('archive', 'archive.zip', 'graphify-out', '.pytest_cache', '.claude', 'docs') }
Compress-Archive -Path $items.FullName -DestinationPath "$HOME\Desktop\traffic_app_submission.zip" -Force
```
