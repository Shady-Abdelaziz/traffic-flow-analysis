# Demo

Results you can view without installing anything.

## Annotated videos

Every vehicle is boxed with its tracking ID, type and speed.

| Video | Traffic |
|---|---|
| [`light_cctv052x2004080516x01640.mp4`](light_cctv052x2004080516x01640.mp4) | light |
| [`medium_cctv052x2004080516x01638.mp4`](medium_cctv052x2004080516x01638.mp4) | medium |
| [`heavy_cctv052x2004080516x01646.mp4`](heavy_cctv052x2004080516x01646.mp4) | heavy |

The videos are MP4 (mp4v). Download them and open them in VLC or Windows Media Player.

[`what_the_shapes_are.png`](what_the_shapes_are.png) shows the busiest frame of the light and
heavy clips, raw beside analysed.

## Charts over all 254 clips

| Chart | Shows |
|---|---|
| [`fundamental_diagram.png`](fundamental_diagram.png) | speed against density, with the fitted traffic-flow curve |
| [`hour_profile.png`](hour_profile.png) | speed and density by hour of day |
| [`lane_usage.png`](lane_usage.png) | occupancy of each lane |
| [`speed_distribution.png`](speed_distribution.png) | speeds per traffic class against the posted limit |

## Colab notebook results

[`notebooks/results.md`](notebooks/results.md) collects the saved figures and printed results
of the calibration and classifier-training notebooks.

## How these files are made

| Files | Command | Detector |
|---|---|---|
| videos and `what_the_shapes_are.png` | `python run.py demo` | `yolo26m` at imgsz 1280 |
| charts | `python run.py detect`, then `python run.py all` | `yolo11n` at imgsz 640 |
| `notebooks/` | `python run.py demo` | from the notebooks' saved outputs |

The demo uses the larger `yolo26m` because three clips take only minutes. The charts cover
all 254 clips, so they use the faster `yolo11n`. See
[Models and hardware](../README.md#5-models-and-hardware).
