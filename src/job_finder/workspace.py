from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "SERPAPI_API_KEY",
    "RX_RESUME_API_KEY",
    "RX_RESUME_API_URL",
)


@dataclass(frozen=True)
class SavedResume:
    name: str
    path: Path


class LocalWorkspace:
    def __init__(
        self,
        *,
        env_path: Path | str = ".env",
        resume_dir: Path | str = ".resume",
        saved_jobs_db_path: Path | str = ".saved_jobs.sqlite3",
    ) -> None:
        self.env_path = Path(env_path)
        self.resume_dir = Path(resume_dir)
        self.saved_jobs_db_path = Path(saved_jobs_db_path)

    def env_exists(self) -> bool:
        return self.env_path.exists()

    def load_env_values(self) -> dict[str, str]:
        if not self.env_path.exists():
            return {}

        values: dict[str, str] = {}
        for raw_line in self.env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    def resolve_value(self, key: str, explicit: str = "") -> str:
        if explicit.strip():
            return explicit.strip()
        if os.getenv(key, "").strip():
            return os.getenv(key, "").strip()
        return self.load_env_values().get(key, "").strip()

    def save_env_values(self, values: dict[str, str]) -> dict[str, str]:
        merged = self.load_env_values()
        for key, value in values.items():
            cleaned = str(value or "").strip()
            if cleaned:
                merged[key] = cleaned
        lines = [f"{key}={merged[key]}" for key in ENV_KEYS if merged.get(key)]
        self.env_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return merged

    def save_uploaded_resume(self, uploaded_path: str | Path) -> SavedResume:
        source = Path(uploaded_path)
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        destination = self.resume_dir / source.name
        shutil.copy2(source, destination)
        return SavedResume(name=destination.name, path=destination)

    def list_saved_resumes(self) -> list[SavedResume]:
        if not self.resume_dir.exists():
            return []
        resumes = [
            SavedResume(name=path.name, path=path)
            for path in sorted(self.resume_dir.glob("*.pdf"))
            if path.is_file()
        ]
        return resumes

    def get_saved_resume(self, name: str) -> SavedResume | None:
        cleaned = str(name or "").strip()
        if not cleaned:
            return None
        path = self.resume_dir / cleaned
        if not path.exists():
            return None
        return SavedResume(name=path.name, path=path)
