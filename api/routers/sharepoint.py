"""
SharePoint integration router — placeholder implementation for MVP.

Real integration would use Microsoft Graph API (ms-graph SDK).
For now this returns mock data so the UI can be built and demonstrated.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from api.auth import get_current_user
from open_notebook.database.repository import (
    ensure_record_id,
    repo_query,
    repo_upsert,
)

router = APIRouter(prefix="/sharepoint", tags=["sharepoint"])


class SharePointConnectRequest(BaseModel):
    site_url: str = Field(..., description="SharePoint site URL")
    folder_path: str = Field(..., description="Folder path within the site")


class SharePointConnectionResponse(BaseModel):
    connected: bool
    site_url: Optional[str] = None
    folder_path: Optional[str] = None
    user_id: Optional[str] = None


class SharePointFile(BaseModel):
    id: str
    name: str
    path: str
    size: int
    modified: str
    file_type: str


class SharePointFolder(BaseModel):
    id: str
    name: str
    path: str
    children: List[Any] = Field(default_factory=list)


class ImportRequest(BaseModel):
    file_ids: List[str] = Field(..., description="IDs of files to import")
    notebook_id: str = Field(..., description="Target notebook ID")


MOCK_FILE_TREE = [
    SharePointFolder(
        id="folder-1",
        name="Research Documents",
        path="/Research Documents",
        children=[
            SharePointFile(
                id="file-1",
                name="Q1 Analysis Report.pdf",
                path="/Research Documents/Q1 Analysis Report.pdf",
                size=245760,
                modified="2026-03-15T10:30:00Z",
                file_type="pdf",
            ),
            SharePointFile(
                id="file-2",
                name="Market Overview 2026.docx",
                path="/Research Documents/Market Overview 2026.docx",
                size=89600,
                modified="2026-04-01T14:22:00Z",
                file_type="docx",
            ),
            SharePointFile(
                id="file-3",
                name="Competitive Landscape.pptx",
                path="/Research Documents/Competitive Landscape.pptx",
                size=512000,
                modified="2026-04-10T09:15:00Z",
                file_type="pptx",
            ),
        ],
    ),
    SharePointFolder(
        id="folder-2",
        name="Team Notes",
        path="/Team Notes",
        children=[
            SharePointFile(
                id="file-4",
                name="Meeting Notes April.docx",
                path="/Team Notes/Meeting Notes April.docx",
                size=34560,
                modified="2026-04-20T16:45:00Z",
                file_type="docx",
            ),
            SharePointFile(
                id="file-5",
                name="Project Roadmap.xlsx",
                path="/Team Notes/Project Roadmap.xlsx",
                size=78200,
                modified="2026-04-18T11:00:00Z",
                file_type="xlsx",
            ),
        ],
    ),
]


def _user_record_id(user_id: str) -> Any:
    return ensure_record_id(user_id if user_id.startswith("user:") else f"user:{user_id}")


async def _get_connection(user_id: str):
    result = await repo_query(
        "SELECT * FROM sharepoint_connection WHERE user_id = $uid LIMIT 1",
        {"uid": _user_record_id(user_id)},
    )
    return result[0] if result else None


@router.get("/connection", response_model=SharePointConnectionResponse)
async def get_connection(current_user: dict = Depends(get_current_user)):
    """Get the current user's SharePoint connection status."""
    conn = await _get_connection(current_user["id"])
    if not conn:
        return SharePointConnectionResponse(connected=False)
    return SharePointConnectionResponse(
        connected=conn.get("connected", False),
        site_url=conn.get("site_url"),
        folder_path=conn.get("folder_path"),
        user_id=current_user["id"],
    )


@router.post("/connect", response_model=SharePointConnectionResponse)
async def connect(
    request: SharePointConnectRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save SharePoint connection settings for the current user (placeholder — no real OAuth)."""
    uid = _user_record_id(current_user["id"])

    existing = await _get_connection(current_user["id"])
    record_id = existing["id"] if existing else None

    data = {
        "user_id": uid,
        "site_url": request.site_url,
        "folder_path": request.folder_path,
        "connected": True,
    }

    if record_id:
        await repo_upsert("sharepoint_connection", str(record_id), data)
    else:
        from open_notebook.database.repository import repo_create
        await repo_create("sharepoint_connection", data)

    logger.info(f"SharePoint connection saved for user {current_user['id']} (placeholder)")

    return SharePointConnectionResponse(
        connected=True,
        site_url=request.site_url,
        folder_path=request.folder_path,
        user_id=current_user["id"],
    )


@router.delete("/connection")
async def disconnect(current_user: dict = Depends(get_current_user)):
    """Disconnect SharePoint for the current user."""
    conn = await _get_connection(current_user["id"])
    if conn:
        await repo_upsert(
            "sharepoint_connection",
            str(conn["id"]),
            {"connected": False, "site_url": None, "folder_path": None},
            add_timestamp=False,
        )
    return {"message": "SharePoint disconnected"}


@router.get("/browse")
async def browse_files(
    path: str = "/",
    _current_user: dict = Depends(get_current_user),
):
    """
    Browse SharePoint folder contents.
    PLACEHOLDER: returns mock file/folder data.
    Replace with Microsoft Graph API call: GET /sites/{site-id}/drive/root:/{path}:/children
    """
    conn = await _get_connection(_current_user["id"])
    if not conn or not conn.get("connected"):
        raise HTTPException(status_code=400, detail="SharePoint is not connected")

    return {
        "path": path,
        "items": [item.model_dump() for item in MOCK_FILE_TREE],
        "placeholder": True,
        "note": "This is mock data. Real integration requires Microsoft Graph API credentials.",
    }


@router.post("/import")
async def import_files(
    request: ImportRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Import selected SharePoint files as sources in a notebook.
    PLACEHOLDER: simulates import without actual file download.
    Real implementation: download via Graph API then call source processing pipeline.
    """
    conn = await _get_connection(current_user["id"])
    if not conn or not conn.get("connected"):
        raise HTTPException(status_code=400, detail="SharePoint is not connected")

    imported = []
    for file_id in request.file_ids:
        imported.append({
            "file_id": file_id,
            "status": "queued",
            "message": "Import queued (placeholder — no actual download yet)",
        })

    logger.info(
        f"SharePoint import requested for {len(request.file_ids)} files "
        f"into notebook {request.notebook_id} by user {current_user['id']} [placeholder]"
    )

    return {
        "imported": imported,
        "notebook_id": request.notebook_id,
        "placeholder": True,
        "note": "Real import requires Microsoft Graph API integration.",
    }
