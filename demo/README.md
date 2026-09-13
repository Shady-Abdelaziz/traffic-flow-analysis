# Demo

| File | What it shows |
|---|---|
| `light_cctv052x2004080516x01640.mp4` | annotated light-traffic clip: boxes, ids, speeds, measurement zone |
| `medium_cctv052x2004080516x01638.mp4` | the same for medium traffic |
| `heavy_cctv052x2004080516x01646.mp4` | the same for heavy traffic |
| `fundamental_diagram.png` | speed against density over all 254 clips, with the fitted curve |
| `hour_profile.png` | level of service by hour of day |
| `lane_usage.png` | lane occupancy |
| `speed_distribution.png` | speeds per class against the posted limit |
| `what_the_shapes_are.png` | two frames before and after annotation: the measurement zone, and each vehicle's box with id, type and speed |
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
   UniDepth → focal length. Check that used no speeds: free-flow 93.2 km/h vs 96.6 posted (−3%).
3. **Live demo.** YOLO + ByteTrack every frame; every second `pipeline.measure` — the same
   function the batch uses — gives speed (RANSAC fit), vehicles observed (≥ 5 frames in zone),
   density → LOS; the CNN classifies 3 frames 1 s apart; the road turns clay where it jams.
4. **Results.** Critical density 38 pc/mi/ln, capacity 1767 veh/h/ln, congestion 15:00–19:00 →
   ramp metering.
5. **Accuracy.** 94.5% / 0.903 macro-F1 with whole recordings held out, against a 65% floor and a
   0.584 no-pixels bar.
6. **Limits.** 48% focal-estimate spread, lane widths not measured, 5 s clips, ID switches.

## Submission zip (without the dataset)

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
$items = Get-ChildItem -Force | Where-Object { $_.Name -notin @('archive', 'archive.zip', 'graphify-out', '.pytest_cache', '.claude', 'docs') }
Compress-Archive -Path $items.FullName -DestinationPath "$HOME\Desktop\traffic_app_submission.zip" -Force
```
