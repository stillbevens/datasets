"""Minimal private datasets hub server.

This module provides a small FastAPI application that implements a subset of the
Hugging Face Hub behaviour required by the :mod:`datasets` library. It supports
creating dataset repositories, uploading files, listing repositories, and
retrieving stored assets. Files are persisted on the local filesystem which
makes the server ideal for on-premises deployments or testing environments.

The service intentionally keeps the API surface small while matching the URL
layout expected by :func:`datasets.load_dataset` when ``HF_ENDPOINT`` points to
this server::

    GET  /datasets                                -> list repositories
    POST /datasets                                -> create a repository
    GET  /datasets/{repo_id}                      -> fetch repository metadata
    POST /datasets/{repo_id}/upload               -> upload a file blob
    GET  /datasets/{repo_id}/resolve/{rev}/{path} -> download a file blob

Because the server is meant for private installations the authentication
mechanism is left to the surrounding infrastructure (for example Nginx).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

HOST_ENV = "DATASETS_PRIVATE_HOST"
PORT_ENV = "DATASETS_PRIVATE_PORT"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

STORAGE_ROOT_ENV = "DATASETS_PRIVATE_STORAGE"
DEFAULT_STORAGE_ROOT = Path.cwd() / "private_hub_store"
METADATA_FILENAME = "metadata.json"
REVISION_DEFAULT = "main"

app = FastAPI(title="Datasets Private Hub", version="0.1.0")


class DatasetCreate(BaseModel):
    """Payload for repository creation."""

    repo_id: str = Field(..., description="Unique repository identifier.")
    description: Optional[str] = Field(default=None, description="Optional description for the dataset.")


class DatasetInfo(BaseModel):
    """Response schema describing a dataset repository."""

    repo_id: str
    description: Optional[str]
    revisions: Dict[str, int] = Field(default_factory=dict, description="Mapping of revision name to file count.")


class DatasetList(BaseModel):
    """Wrapper used for the list API."""

    datasets: List[DatasetInfo]


_storage_root: Optional[Path] = None


def get_storage_root() -> Path:
    """Return the configured storage directory, creating it if necessary."""

    global _storage_root
    if _storage_root is None:
        root_value = os.getenv(STORAGE_ROOT_ENV)
        _storage_root = Path(root_value) if root_value else DEFAULT_STORAGE_ROOT
    root = _storage_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_storage_root(path: Path | str) -> Path:
    """Override the storage directory at runtime."""

    global _storage_root
    _storage_root = Path(path)
    return get_storage_root()


def sanitize_repo_id(repo_id: str) -> str:
    cleaned = repo_id.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Repository ID cannot be empty")
    if any(part in {"", ".", ".."} for part in cleaned.split("/")):
        raise HTTPException(status_code=400, detail="Repository ID contains invalid segments")
    return cleaned


def repo_path(repo_id: str) -> Path:
    safe_id = sanitize_repo_id(repo_id).replace("/", "__")
    return get_storage_root() / safe_id


def revision_path(repo_id: str, revision: str) -> Path:
    clean_revision = revision.strip() or REVISION_DEFAULT
    if any(part in {"", ".", ".."} for part in clean_revision.split("/")):
        raise HTTPException(status_code=400, detail="Invalid revision name")
    return repo_path(repo_id) / clean_revision


def metadata_path(repo_id: str) -> Path:
    return repo_path(repo_id) / METADATA_FILENAME


def load_metadata(repo_id: str) -> Dict[str, object]:
    path = metadata_path(repo_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metadata(repo_id: str, payload: Dict[str, object]) -> None:
    path = metadata_path(repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


@app.get("/datasets", response_model=DatasetList)
async def list_datasets() -> DatasetList:
    datasets: List[DatasetInfo] = []
    root = get_storage_root()
    for repo_dir in root.glob("*"):
        if not repo_dir.is_dir():
            continue
        metadata_file = repo_dir / METADATA_FILENAME
        if not metadata_file.exists():
            continue
        with metadata_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        revisions = metadata.get("revisions", {})
        datasets.append(
            DatasetInfo(
                repo_id=metadata.get("repo_id", repo_dir.name.replace("__", "/")),
                description=metadata.get("description"),
                revisions={name: count for name, count in revisions.items()},
            )
        )
    return DatasetList(datasets=datasets)


@app.post("/datasets", response_model=DatasetInfo, status_code=201)
async def create_dataset(payload: DatasetCreate) -> DatasetInfo:
    repo_dir = repo_path(payload.repo_id)
    if repo_dir.exists():
        raise HTTPException(status_code=409, detail="Dataset already exists")
    repo_dir.mkdir(parents=True, exist_ok=True)
    info = DatasetInfo(repo_id=payload.repo_id, description=payload.description, revisions={REVISION_DEFAULT: 0})
    info_dict = info.model_dump() if hasattr(info, "model_dump") else info.dict()
    save_metadata(payload.repo_id, info_dict)
    return info


@app.get("/datasets/{repo_id}", response_model=DatasetInfo)
async def get_dataset(repo_id: str) -> DatasetInfo:
    metadata = load_metadata(repo_id)
    return DatasetInfo(**metadata)


@app.post("/datasets/{repo_id}/upload", response_model=DatasetInfo)
async def upload_file(repo_id: str, file: UploadFile = File(...), revision: str = REVISION_DEFAULT) -> DatasetInfo:
    repo_dir = repo_path(repo_id)
    if not repo_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    target_dir = revision_path(repo_id, revision)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / file.filename
    with target_file.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    metadata = load_metadata(repo_id)
    revisions = metadata.setdefault("revisions", {})
    revisions[revision] = len(list(target_dir.rglob("*")))
    save_metadata(repo_id, metadata)
    return DatasetInfo(**metadata)


@app.get("/datasets/{repo_id}/resolve/{revision}/{file_path:path}")
async def download_file(repo_id: str, revision: str, file_path: str):
    target_dir = revision_path(repo_id, revision)
    target_file = target_dir / file_path
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Requested file not found")
    return FileResponse(target_file)


@app.delete("/datasets/{repo_id}", status_code=204)
async def delete_dataset(repo_id: str):
    repo_dir = repo_path(repo_id)
    if not repo_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    shutil.rmtree(repo_dir)
    return JSONResponse(status_code=204, content=None)


def main(argv: Optional[List[str]] = None) -> None:
    """Launch an HTTP server serving the private hub."""

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the datasets private hub server.")
    parser.add_argument(
        "--storage-root",
        help=(
            "Directory used to persist uploaded datasets. Overrides the "
            f"{STORAGE_ROOT_ENV} environment variable and defaults to"
            f" {DEFAULT_STORAGE_ROOT}"
        ),
    )
    parser.add_argument(
        "--host",
        help=(
            "Host interface to bind. Overrides the "
            f"{HOST_ENV} environment variable and defaults to {DEFAULT_HOST}"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Port to bind. Overrides the "
            f"{PORT_ENV} environment variable and defaults to {DEFAULT_PORT}"
        ),
    )
    args = parser.parse_args(argv)

    if args.storage_root:
        set_storage_root(args.storage_root)

    host = args.host or os.getenv(HOST_ENV, DEFAULT_HOST)
    port_value = args.port or os.getenv(PORT_ENV)
    port = int(port_value) if port_value is not None else DEFAULT_PORT

    uvicorn.run(app, host=host, port=port)


__all__ = ["app", "get_storage_root", "main", "set_storage_root"]
