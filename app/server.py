"""The website's API: it starts live sessions and measures the saved detections.

    python run.py serve        # http://127.0.0.1:8000

One live session at a time: detection is CPU-bound, so a new one stops the previous one.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.live import LiveSession, Models, probe_video
from trafficflow.config import Paths, load_calibration, load_site
from trafficflow.dataset import TrafficDatabase
from trafficflow.detector import DetectorConfig
from trafficflow.tracks import stale_fields
from trafficflow.insights import build_results
from trafficflow.pipeline import analyse_all, analyse_uploads

__all__ = ["create_app"]

STATIC = Path(__file__).parent / "static"


def create_app(paths: Paths | None = None, models: Models | None = None) -> FastAPI:
    where = paths or Paths()
    site = load_site(where.config / "site.yaml")
    calibration = load_calibration(where.calibration)
    db = TrafficDatabase.load(where.archive)
    loaded = models or Models.load()
    #: Uploads used to live in a temp folder that was deleted on shutdown, so nothing
    #: about an uploaded video survived and it could never reach the Results page. They
    #: are kept now, one folder each, and removed only when asked.
    uploads = where.uploads
    uploads.mkdir(parents=True, exist_ok=True)
    sessions: dict[str, LiveSession] = {}
    #: Serialises everything that swaps the running session. Detection is CPU-bound and
    #: every session shares the one loaded YOLO model, so two overlapping POSTs could
    #: otherwise leave two workers stepping through the same tracker.
    starting = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        # Shutting down: stop whatever is still analysing. The uploads are kept -- they
        # are the subject of the Results page now, not scratch files.
        for running in sessions.values():
            running.stop()

    app = FastAPI(title="Traffic Flow", version="4.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def session_or_404(session_id: str) -> LiveSession:
        if session_id not in sessions:
            raise HTTPException(404, "no such session -- start the video again")
        return sessions[session_id]

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/clips", summary="The 254 dataset clips, for the picker")
    def clips() -> list[dict]:
        current = DetectorConfig()

        def usable(clip: str) -> bool:
            table = where.tracks / f"{clip}.csv"
            return table.exists() and not stale_fields(table, current)

        return [{"name": c.name, "file": c.video_path(db.data_root).name,
                 "traffic_class": c.traffic_class, "cached": usable(c.name)}
                for c in db.clips]

    def forget_sessions(clip: str | None = None) -> None:
        """Stop and drop the in-memory sessions for one clip, or all of them."""
        for session_id, session in list(sessions.items()):
            if clip is None or session.name == clip:
                session.stop()
                del sessions[session_id]

    @app.delete("/api/cache/{clip}", summary="Delete one clip's saved tracks; its video is kept")
    def remove_cache(clip: str) -> dict:
        try:
            db[clip]
        except KeyError:
            raise HTTPException(404, f"no clip called {clip}")
        forget_sessions(clip)
        saved = where.tracks / f"{clip}.csv"
        removed = saved.exists()
        saved.unlink(missing_ok=True)
        return {"removed": int(removed)}

    @app.delete("/api/cache", summary="Remove every video the app knows: saved detections and uploads")
    def clear_cache() -> dict:
        """Empties what the website shows -- its detections and its uploaded videos.

        Deliberately narrow. It does **not** touch the batch run's artefacts in
        ``output/videos``, ``output/figures``, ``output/reports`` or
        ``output/parameters``: those are the project's deliverables, they cost a
        full detection run to rebuild, and clearing the app's videos is not a
        reason to destroy them. The dataset in ``archive/`` is never touched either.
        """
        forget_sessions()
        saved = list(where.tracks.glob("*.csv")) if where.tracks.exists() else []
        for path in saved:
            path.unlink(missing_ok=True)
        folders = [f for f in uploads.iterdir() if f.is_dir()] if uploads.exists() else []
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
        return {"removed": len(saved) + len(folders), "uploads_removed": len(folders)}

    def upload_folder(upload_id: str) -> Path:
        """The folder for one upload, refusing anything that is not a plain id."""
        if not upload_id or set(upload_id) - set("0123456789abcdef"):
            raise HTTPException(404, f"no upload called {upload_id}")
        folder = uploads / upload_id
        if not folder.is_dir():
            raise HTTPException(404, f"no upload called {upload_id}")
        return folder

    def upload_meta(folder: Path) -> dict | None:
        meta = folder / "meta.json"
        if not meta.exists():
            return None
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None            # a half-written file, from a crash mid-upload

    def stored_uploads() -> list[dict]:
        """Every kept upload, newest first, with whether its detection was saved."""
        found = []
        for folder in uploads.iterdir():
            if not folder.is_dir():
                continue
            meta = upload_meta(folder)
            if meta is None:
                continue
            found.append({"id": folder.name, "name": meta.get("name", folder.name),
                          "uploaded_at": meta.get("uploaded_at", ""),
                          "cached": (folder / "tracks.csv").exists(),
                          "traffic_class": meta.get("traffic_class")})
        return sorted(found, key=lambda item: item["uploaded_at"], reverse=True)

    def store_upload(file: UploadFile) -> tuple[Path, Path, str]:
        """Write one uploaded video into its own folder, with the metadata to find it.

        Returns ``(folder, video, name)``. A uuid names the folder so two uploads sharing
        a filename cannot clobber each other, while the name the user recognises is kept
        in the metadata.

        The metadata is written here rather than when detection ends, so an upload that
        is never analysed -- or whose analysis is stopped part-way -- is still listed and
        still removable. :func:`~trafficflow.pipeline.analyse_uploads` skips a folder
        with metadata and no track table.

        Raises
        ------
        HTTPException
            400 if the file cannot be decoded as a video; nothing is left on disk.
        """
        name = Path(file.filename or "upload.mp4").name
        folder = uploads / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / name
        try:
            with video.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            fps, frames = probe_video(video, site.image.fps)
        except (ValueError, OSError) as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(400, str(error))

        (folder / "meta.json").write_text(json.dumps({
            "id": folder.name, "name": name, "file": video.name,
            "fps": fps, "frames": frames,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        return folder, video, name

    @app.get("/api/uploads", summary="The videos uploaded to this server, newest first")
    def uploaded() -> list[dict]:
        return stored_uploads()

    @app.post("/api/uploads", summary="Store a video without analysing it")
    def store(file: UploadFile = File(...)) -> dict:
        """Keep a video for later, and say nothing about analysing it.

        The Live page takes several files at once and analyses them one at a time, so it
        stores the whole selection first. Posting them all to ``/api/live`` instead would
        start a session per file and each would stop the one before it.
        """
        folder, _video, name = store_upload(file)
        return {"id": folder.name, "name": name}

    @app.delete("/api/uploads", summary="Delete every upload; saved dataset detections are kept")
    def clear_uploads() -> dict:
        """What the page's Clear button calls, and deliberately narrower than ``/api/cache``.

        ``/api/cache`` also deletes the track table of every dataset clip. Those cost a
        long detection pass to rebuild, and the Live page has no dataset picker to
        rebuild them from, so a button that could destroy them by accident has no place
        on it.
        """
        forget_sessions()
        folders = [f for f in uploads.iterdir() if f.is_dir()] if uploads.exists() else []
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)
        return {"removed": len(folders)}

    @app.delete("/api/uploads/{upload_id}", summary="Delete one upload: its video, tracks and metadata")
    def remove_upload(upload_id: str) -> dict:
        folder = upload_folder(upload_id)
        meta = upload_meta(folder)
        # Without readable metadata the session playing it cannot be identified by name,
        # so every session is dropped rather than leaving one reading a deleted file.
        forget_sessions(meta.get("name") if meta else None)
        shutil.rmtree(folder, ignore_errors=True)
        return {"removed": 1}

    @app.post("/api/live", summary="Analyse a dataset clip, a kept upload, or a new video file")
    def start(clip: str | None = Form(None), upload: str | None = Form(None),
              file: UploadFile | None = File(None)) -> dict:
        chosen = [given is not None for given in (clip, upload, file)]
        if sum(chosen) != 1:
            raise HTTPException(400, "send exactly one of: a clip name, an upload id, or a video file")

        with starting:
            tracks, folder = None, None
            if clip is not None:
                try:
                    video, name = db[clip].video_path(db.data_root), clip
                except KeyError:
                    raise HTTPException(404, f"no clip called {clip}")
                tracks = where.tracks / f"{clip}.csv"   # reused when saved, written after a fresh detection
            elif upload is not None:
                folder = upload_folder(upload)
                meta = upload_meta(folder)
                video = folder / (meta or {}).get("file", "")
                if meta is None or not video.exists():
                    raise HTTPException(404, f"upload {upload} has no video left to play")
                name, tracks = meta.get("name", upload), folder / "tracks.csv"
            else:
                assert file is not None                 # guaranteed by the check above
                folder, video, name = store_upload(file)   # rejects a non-video with a 400
                tracks = folder / "tracks.csv"

            # Validate before touching the running session: a file that turns out not
            # to be a video used to stop whatever was playing and then return a 400,
            # so a bad upload killed a good session.
            try:
                session = LiveSession(video, name, site, calibration, loaded, tracks=tracks,
                                      uploaded=file is not None or upload is not None)
            except ValueError as error:
                if file is not None and folder is not None:
                    shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(400, str(error))

            for running in sessions.values():
                running.stop()                          # sets the flag *and* joins
            sessions.clear()
            sessions[session.id] = session
            session.start()
            return session.info()

    @app.get("/api/live/{session_id}/events", summary="Server-sent events: progress, stats, frame, end, error")
    def events(session_id: str):
        session = session_or_404(session_id)

        def stream():
            sent = 0
            while True:
                batch = session.events_after(sent)
                if not batch:
                    yield ": still working\n\n"
                    continue
                for kind, data in batch:
                    sent += 1
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    if kind in ("end", "error"):
                        return

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.get("/api/live/{session_id}/frames/{number}.jpg", summary="One annotated frame")
    def frame(session_id: str, number: int):
        jpeg = session_or_404(session_id).jpegs.get(number)
        if jpeg is None:
            raise HTTPException(404, "that frame has not been analysed")
        return Response(jpeg, media_type="image/jpeg")

    @app.post("/api/live/{session_id}/stop", summary="Stop analysing")
    def stop(session_id: str) -> dict:
        session = session_or_404(session_id)
        session.stop()
        return {"state": session.state}

    def tracks_state() -> tuple[tuple[str, int], ...]:
        """Every saved table and when it was written; any save or delete changes this.

        Covers the uploads as well as the dataset, so analysing or removing an uploaded
        video invalidates the cached answer exactly as a dataset clip does.
        """
        found = []
        if where.tracks.exists():
            found += [(p.name, p.stat().st_mtime_ns) for p in where.tracks.glob("*.csv")]
        if uploads.exists():
            found += [(str(p.relative_to(uploads)), p.stat().st_mtime_ns)
                      for p in uploads.glob("*/tracks.csv")]
        return tuple(sorted(found))

    #: The last answer and the disk state it was computed from, so repeat visits are free.
    last: dict = {}

    @app.get("/api/results", summary="Findings over the clips that have saved detections")
    def results() -> dict:
        """Measured fresh from ``output/tracks/`` so the page follows what is saved.

        Reading a precomputed table instead meant a deleted detection stayed on this page
        with its old numbers, and a clip analysed in the browser never reached it.
        """
        state = tracks_state()
        if last.get("state") != state:
            measured = analyse_all(db, site, calibration, where) + analyse_uploads(site, calibration, where)
            if not measured:
                raise HTTPException(
                    404, "No saved detections yet. Upload a video, play a clip, "
                         "or run `python run.py detect`.")
            report = where.models / "eval_finetune.json"
            scores = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
            last.clear()
            last["state"], last["results"] = state, build_results(
                [clip.to_row() for clip in measured], site, scores)
        return last["results"]

    return app
