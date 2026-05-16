from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=os.environ.get("API_RELOAD", "false").lower() == "true",
    )
