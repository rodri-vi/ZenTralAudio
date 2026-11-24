---

`core/__init__.py`
```python
"""Núcleo de infraestructura de ZenTralAudio."""

from core.configuration import AppSettings, ConfigManager
from core.exceptions import ZenTralAudioError
from core.job_manager import CancellationToken, JobHandle, JobManager, JobStatus
from core.paths import AppPaths

__all__ = [
    "AppPaths",
    "AppSettings",
    "CancellationToken",
    "ConfigManager",
    "JobHandle",
    "JobManager",
    "JobStatus",
    "ZenTralAudioError",
]
