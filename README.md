# Traffic Flow Analysis with Computer Vision

A computer vision system that measures traffic from fixed-camera highway video, grades
congestion, and turns the measurements into insights and recommendations. It runs in two
ways: as a batch pipeline over the full dataset, and as a web app that analyses any video live.

**Dataset.** WSDOT / UCSD highway traffic database: Interstate 5 southbound at S 188th St,
Seattle. 254 clips from August 2004, each 320×240 at 10 fps and about 5 seconds long, labelled
*light*, *medium* or *heavy*.

---

## Requirements coverage

| Exam requirement | Where it is met |
|---|---|
| Computer vision API / library | Ultralytics YOLO + ByteTrack, OpenCV, ONNX Runtime |
| Vehicle count and classification | [`counting.py`](trafficflow/counting.py): car, motorcycle, bus, truck |
| Speed estimation | [`speed.py`](trafficflow/speed.py): calibrated ground-plane distance, RANSAC fit |
| Density and congestion level | [`density.py`](trafficflow/density.py) (HCM level of service), [`congestion.py`](trafficflow/congestion.py) (CNN) |
| Lane occupancy and changes | [`lanes.py`](trafficflow/lanes.py) |
| Trajectory analysis | [`trajectory.py`](trafficflow/trajectory.py) |
| Visualization | annotated videos, figures, web dashboard ([`demo/`](demo/)) |
| Insights and recommendations | `output/reports/insights.md` and the Results tab in the web app |

## Deliverables

| Item | Location |
|---|---|
| Python code | `run.py`, `trafficflow/`, `app/` |
| Documentation | this README, [`demo/README.md`](demo/README.md), docstrings |
| Demo and output examples | `demo/`, `output/` |
| Models | `models/` |
| Dependencies | `requirements.txt` |

---

## Quick start

Python 3.11+ on CPU. Install PyTorch from the CPU index first; the default wheels add
about 2.5 GB of CUDA libraries.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset)
and extract it to `archive/` (the `video/` folder and the index files as shipped). Then run:

```bash
python run.py detect          # detect and track vehicles in all 254 clips (resumable)
python run.py all --videos    # parameters, classifier, figures, annotated videos, report
python run.py serve           # web app at http://127.0.0.1:8000  (or double-click start.bat)
```

`detect` is the only slow stage. It takes about 20–30 minutes on a laptop CPU with
`yolo11n` and writes one track table per clip to `output/tracks/`. Every later stage reads
those tables and finishes in seconds.

| Command | Output |
|---|---|
| `python run.py params` | `output/parameters/clips.csv`: every parameter for every clip |
| `python run.py train` | `output/reports/eval_parameters.json`: parameter classifier vs baselines |
| `python run.py visuals [--videos]` | `output/figures/*.png`, `output/videos/*.mp4` |
| `python run.py report` | `output/reports/insights.md` |
| `python run.py detect --compare-sizes --weights yolo11n.pt yolo26m.pt` | detection recall and speed per model and input size |

---

## Pipeline

```
video ─► YOLO + ByteTrack ─► track tables ─► pipeline.measure ─► parameters ─► figures · report · web app
          (detector.py)      (output/tracks)   (pipeline.py)                    congestion CNN (ONNX)
```

1. **Detection and tracking.** COCO-pretrained YOLO, limited to road vehicles, with ByteTrack
   IDs. Only the southbound roadway is sent to the detector, because the northbound lanes
   visible in the frame would inflate counts.
2. **Pixels to metres.** Each vehicle's ground point (the bottom-centre of its box) is mapped
   to the road plane through a homography measured from the footage.
3. **Parameters.** Computed per clip, and every second during live analysis, by the same
   function, so the report and the web app always agree.
4. **Congestion class.** A fine-tuned ResNet-18 reads three frames taken one second apart.

### Detector

| Checkpoint | Role |
|---|---|
| `yolo26m.pt` | Final detector. More accurate: `yolo11n` misses some clear cars and labels many cars as trucks. |
| `yolo11n.pt` | Used for the batch pass over all 254 clips. About 10× less compute per frame, which keeps the full run under half an hour on a CPU. |

The checkpoint is set in one place, `DetectorConfig.weights` in
[`detector.py`](trafficflow/detector.py). Every track table stores the settings that
produced it in a `.json` next to it. A table made with different settings is treated as
stale and detected again, so results from the two checkpoints are never mixed.

### How each parameter is computed

| Parameter | Method |
|---|---|
| Vehicle count and type | A track counts once it spends ≥ 5 frames in the measurement zone. Its type is the majority COCO class over its frames. |
| Speed | RANSAC line through distance along the road against time, over ≥ 10 frames. Fits that are not reliable get no speed label. |
| Density | Passenger-car equivalents in the zone (bus/truck = 2) ÷ (zone length × 5 lanes), in pc/mi/ln |
| Level of service | HCM basic freeway thresholds: A ≤ 11 · B ≤ 18 · C ≤ 26 · D ≤ 35 · E ≤ 45 · F > 45 |
| Lane occupancy and changes | Share of all frames each lane is occupied. Lane changes use lateral position, with debounce and hysteresis. |
| Trajectories | Bird's-eye paths, time–space series, merge detection |

### Camera calibration

A single camera cannot measure distance by itself, so
[`01_calibration.ipynb`](notebooks/01_calibration.ipynb) (Colab) takes the scale from the
footage. The painted road edges give the horizon. The road width (5 × 3.63 m) gives the
camera height. UniDepth v2 gives the focal length. A quadratic fit follows the road's curve.
The result is stored in `config/calibration.yaml`. No vehicle speed is used in calibration,
so the free-flow speed from the fundamental diagram, compared with the posted 96.6 km/h,
serves as an independent check.

---

## Notebooks (run on Google Colab)

The two GPU stages were run on **Google Colab**, not locally. Each notebook is saved with its
Colab outputs, so every figure and score can be inspected without running it again.

| Notebook | What it does | Files it produces |
|---|---|---|
| [`01_calibration.ipynb`](notebooks/01_calibration.ipynb) | Measures the camera: horizon, height, focal length, road curve | `config/calibration.yaml` |
| [`02_finetune.ipynb`](notebooks/02_finetune.ipynb) | Fine-tunes ResNet-18 and scores it with recordings held out | `models/congestion_cnn.onnx`, `models/eval_finetune.json` |

Both notebooks are self-contained. They import nothing from this repository and download
the dataset themselves through `kagglehub`. To reproduce a stage:

1. Open the notebook in Colab.
2. Select a GPU runtime (Runtime → Change runtime type).
3. Run all cells.
4. Put the downloaded files at the paths listed in the table above.

The local pipeline only reads these files. It never needs a GPU.

---

## Congestion classifier

Fine-tuned ResNet-18 ([`02_finetune.ipynb`](notebooks/02_finetune.ipynb), Colab). It is
exported to ONNX with its preprocessing embedded, so local inference cannot drift from training.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always "light" (floor) | 65.0% | 0.263 |
| Nearest recording, no pixels (bar) | 71.3% | 0.584 |
| **Fine-tuned ResNet-18** | **94.5%** | **0.903** |

Scores come from cross-validation over all 254 clips, with **whole recordings held out**.
The archive's own folds put near-duplicate clips on both sides of the split. Per-class
recall: light 99%, medium 82%, heavy 89%. The full results are in `models/eval_finetune.json`.

---

## Web app

`python run.py serve` opens two tabs:

- **Live.** Pick a dataset clip or drop in any short video. Each vehicle gets a box, ID, type
  and speed. Every second the panel updates congestion class, level of service, median speed,
  density, vehicle count, lane occupancy and a one-line insight. Playback has a timeline and
  an FPS control.
- **Results.** Findings across every analysed video: free-flow speed, capacity, critical
  density, the speed–density chart, speed by hour, classifier accuracy and recommendations.

Uploads are resized to 320×240 and use this camera's calibration. On footage from another
camera, counts and vehicle types still hold, but speed, density and congestion class do not.

### API

Interactive reference at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/clips` | Dataset clips, and whether their detections are cached |
| POST | `/api/live` | Start an analysis. Form field `clip`, `upload` **or** `file` (exactly one). |
| GET | `/api/live/{id}/events` | Server-sent events: `progress`, `stats` (every second), `frame`, `end`, `error` |
| GET | `/api/live/{id}/frames/{n}.jpg` | Annotated frame *n* |
| POST | `/api/live/{id}/stop` | Stop an analysis |
| GET · POST · DELETE | `/api/uploads`, `/api/uploads/{id}` | List, store or delete uploaded videos |
| DELETE | `/api/cache`, `/api/cache/{clip}` | Delete saved detections (the dataset is never touched) |
| GET | `/api/results` | Everything the Results tab shows |

```bash
curl -F clip=cctv052x2004080516x01646 http://127.0.0.1:8000/api/live
curl -N http://127.0.0.1:8000/api/live/<id>/events
```

---

## Project structure

```
run.py                 command-line entry point for every stage
start.bat              double-click to open the web app
requirements.txt
models/                yolo26m.pt · yolo11n.pt · congestion_cnn.onnx · eval_finetune.json
notebooks/             01_calibration.ipynb · 02_finetune.ipynb   (Colab, GPU)
config/                site.yaml (road facts, with sources) · calibration.yaml (generated)
trafficflow/           analysis package, one module per job
  detector.py            YOLO + ByteTrack
  counting.py · speed.py · density.py · lanes.py · trajectory.py
  pipeline.py            tracks → all parameters (shared by batch and web app)
  congestion.py          CNN inference
  charts.py · overlay.py · insights.py
app/                   FastAPI server and static front end (no build step)
tests/                 pytest suite
demo/                  annotated clips, figures, screenshots, demo recording
output/                tracks · parameters · figures · videos · reports · logs
archive/               dataset (not included in the submission)
```

## Testing

```bash
python -m pytest -q
```

No ground-truth speeds come with the dataset, so correctness is checked independently of
the calibration. The checks are: free-flow speed against the posted limit, speeds ordered
light > medium > heavy, a textbook-shaped fundamental diagram, a classifier that beats both
baselines, and the annotated video, where a box on the wrong vehicle is easy to spot.

## Known limitations

- **Short clips.** At about 5 s each, counts are vehicles observed rather than hourly flow,
  and density is a snapshot compared against HCM thresholds defined on 15-minute averages.
- **Input size is not tuned.** Detection runs at 640 px for CPU speed. How many distant
  vehicles it loses compared with 960 has not been measured; `detect --compare-sizes`
  measures it.
- **Lane widths are assumed.** Interior lane lines are not visible at 320×240, so the road
  width is split into five equal lanes. This affects lane assignment only.
- **Focal-length estimates disagree** (793 px from UniDepth intrinsics against 412 px implied by
  its depth map). The intrinsics value is used because the free-flow check agrees with it.
- **ID switches** can count one vehicle twice. **Motorcycle** counts are a lower bound.
- **No turning or merging.** The camera shows an Interstate mainline with no junction in view.
