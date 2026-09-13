"""Traffic flow analysis on the UCSD/WSDOT I-5 highway video database.

The package is a flat set of modules, each one stage of a pipeline that communicates
through files on disk rather than by calling the next stage directly:

``dataset``
    The 254-clip index: metadata, labels, and the published evaluation folds.
``video``
    Frame access, skipping each clip's corrupted first frame.
``geometry``
    The image-to-road-plane mapping. Reads ``config/camera.yaml``, converts image
    points to metres, and assigns vehicles to lanes.
``detection``
    Detector plus tracker over the clips, writing one track table per clip. The only
    expensive stage, so it is never repeated.
``parameters``
    Track tables to traffic parameters: counts and classes, speeds, density and level
    of service, lane occupancy and lane changes, trajectories.
``model``
    Congestion classifiers, scored with the database's own four-fold protocol.
``viz`` / ``insights`` / ``api``
    Figures and annotated video, the written analysis, and the service that serves
    precomputed results.

Because each stage reads the previous stage's output from disk, re-rendering a figure
or changing a threshold costs seconds instead of re-running detection.

Run stages through the ``run.py`` entry point at the repository root::

    python run.py detect
    python run.py params
"""

from __future__ import annotations

from trafficflow.dataset import (
    CONSTANT_COLUMNS,
    TRAFFIC_CLASSES,
    Clip,
    Fold,
    TrafficDatabase,
)
from trafficflow.video import (
    FIRST_USABLE_FRAME,
    VideoInfo,
    iter_frames,
    probe,
    read_frame,
)

__version__ = "0.1.0"

__all__ = [
    "CONSTANT_COLUMNS",
    "TRAFFIC_CLASSES",
    "Clip",
    "Fold",
    "TrafficDatabase",
    "FIRST_USABLE_FRAME",
    "VideoInfo",
    "iter_frames",
    "probe",
    "read_frame",
    "__version__",
]
