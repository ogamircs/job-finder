from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .models import (
    CandidateProfile,
    GeneratedApplicationArtifacts,
    ScoredJobMatch,
    TailoredApplicationContent,
)
from .resume_sources import (
    delete_rxresume_resume,
    download_file,
    export_rxresume_resume_pdf,
    import_rxresume_resume,
    load_rxresume_resume_document,
)

DEFAULT_APPLICATION_DOCS_MODEL = "gpt-4o"


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return ""


def tailored_application_content_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "headline": {"type": "string"},
            "summary": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "keywords"],
                },
            },
            "cover_letter": {"type": "string"},
        },
        "required": ["headline", "summary", "skills", "cover_letter"],
    }


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return cleaned or "artifact"


def _clone_json(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _ensure_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    current = parent.get(key)
    if isinstance(current, dict):
        return current
    next_value: dict[str, Any] = {}
    parent[key] = next_value
    return next_value


def _extract_resume_payload(resume_document: dict[str, Any]) -> dict[str, Any]:
    data = resume_document.get("data")
    if isinstance(data, dict):
        return _clone_json(data)
    return _clone_json(resume_document)


def apply_tailored_resume_content(
    base_resume: dict[str, Any],
    tailored_content: TailoredApplicationContent,
) -> dict[str, Any]:
    resume = _clone_json(base_resume)

    basics = _ensure_dict(resume, "basics")
    sections = _ensure_dict(resume, "sections")
    summary_root = _ensure_dict(resume, "summary")
    summary_section = _ensure_dict(sections, "summary")
    skills_section = _ensure_dict(sections, "skills")

    if tailored_content.headline:
        basics["headline"] = tailored_content.headline
        basics["label"] = tailored_content.headline

    if tailored_content.summary:
        basics["summary"] = tailored_content.summary
        summary_root["content"] = tailored_content.summary
        summary_section["content"] = tailored_content.summary

    if tailored_content.skills:
        existing_items = skills_section.get("items")
        template_items = [item for item in existing_items if isinstance(item, dict)] if isinstance(existing_items, list) else []
        default_template = template_items[0] if template_items else {}
        next_items: list[dict[str, Any]] = []
        for index, skill in enumerate(tailored_content.skills):
            template = dict(template_items[index]) if index < len(template_items) else dict(default_template)
            if "id" not in template or not str(template.get("id") or "").strip():
                template["id"] = f"tailored-skill-{index + 1}"
            if "hidden" not in template:
                template["hidden"] = False
            if "icon" not in template:
                template["icon"] = ""
            if "proficiency" not in template:
                template["proficiency"] = ""
            if "level" not in template:
                template["level"] = 0

            template["name"] = skill.name
            template["keywords"] = list(skill.keywords)
            next_items.append(template)

        skills_section["items"] = next_items

    return resume


def generate_tailored_application_content(
    *,
    base_resume: dict[str, Any],
    scored_job: ScoredJobMatch,
    openai_api_key: str,
    openai_model: str = DEFAULT_APPLICATION_DOCS_MODEL,
    candidate_profile: CandidateProfile | dict[str, Any] | None = None,
    client: Any | None = None,
) -> TailoredApplicationContent:
    openai_client = client or OpenAI(api_key=openai_api_key)
    prompt = {
        "resume": base_resume,
        "candidate_profile": candidate_profile.model_dump()
        if isinstance(candidate_profile, CandidateProfile)
        else (candidate_profile or {}),
        "job": scored_job.model_dump(),
        "instructions": [
            "Tailor the resume to this job without inventing experience.",
            "Rewrite the headline and summary for the target role.",
            "Return 2 to 4 concise skill categories with keywords grounded in the resume and job.",
            "Write a concise cover letter in plain text with no markdown bullets.",
        ],
    }
    response = openai_client.responses.create(
        model=openai_model or DEFAULT_APPLICATION_DOCS_MODEL,
        input=json.dumps(prompt),
        text={
            "format": {
                "type": "json_schema",
                "name": "tailored_application_content",
                "schema": tailored_application_content_schema(),
                "strict": True,
            },
            "verbosity": "medium",
        },
        max_output_tokens=2400,
    )
    payload = json.loads(_response_text(response))
    return TailoredApplicationContent.model_validate(payload)


class ApplicationArtifactsService:
    def __init__(
        self,
        *,
        output_root: Path | str = "output/generated",
        now_provider: Callable[[], datetime] | None = None,
        resume_loader: Callable[..., dict[str, Any]] = load_rxresume_resume_document,
        content_generator: Callable[..., TailoredApplicationContent] = generate_tailored_application_content,
        resume_importer: Callable[..., str] = import_rxresume_resume,
        pdf_exporter: Callable[..., str] = export_rxresume_resume_pdf,
        file_downloader: Callable[..., None] = download_file,
        resume_deleter: Callable[..., None] = delete_rxresume_resume,
    ) -> None:
        self.output_root = Path(output_root)
        self.now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.resume_loader = resume_loader
        self.content_generator = content_generator
        self.resume_importer = resume_importer
        self.pdf_exporter = pdf_exporter
        self.file_downloader = file_downloader
        self.resume_deleter = resume_deleter

    def _artifact_dir(self, scored_job: ScoredJobMatch, generated_at: datetime) -> Path:
        stamp = generated_at.strftime("%Y%m%d-%H%M%S")
        slug = _slugify(f"{scored_job.job.company}-{scored_job.job.title}")
        return self.output_root / f"{stamp}-{slug}"

    def generate_application_artifacts(
        self,
        *,
        rxresume_base_url: str,
        rxresume_api_key: str,
        base_resume_id: str,
        scored_job: ScoredJobMatch,
        openai_api_key: str,
        openai_model: str = DEFAULT_APPLICATION_DOCS_MODEL,
        candidate_profile: CandidateProfile | dict[str, Any] | None = None,
    ) -> GeneratedApplicationArtifacts:
        generated_at = self.now_provider()
        artifact_dir = self._artifact_dir(scored_job, generated_at)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        base_resume_document = self.resume_loader(rxresume_base_url, rxresume_api_key, base_resume_id)
        base_resume_payload = _extract_resume_payload(base_resume_document)

        tailored_content = TailoredApplicationContent.model_validate(
            self.content_generator(
                base_resume=base_resume_payload,
                scored_job=scored_job,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                candidate_profile=candidate_profile,
            )
        )
        tailored_resume = apply_tailored_resume_content(base_resume_payload, tailored_content)

        remote_resume_id: str | None = None
        pdf_url = ""
        try:
            remote_resume_id = self.resume_importer(
                rxresume_base_url,
                rxresume_api_key,
                tailored_resume,
                name=f"{scored_job.job.company} - {scored_job.job.title}",
                slug="",
            )
            pdf_url = self.pdf_exporter(rxresume_base_url, rxresume_api_key, remote_resume_id)

            pdf_path = artifact_dir / "tailored_resume.pdf"
            cover_letter_path = artifact_dir / "cover_letter.md"
            resume_json_path = artifact_dir / "tailored_resume.json"
            metadata_path = artifact_dir / "metadata.json"

            self.file_downloader(pdf_url, pdf_path)
            cover_letter_path.write_text(f"{tailored_content.cover_letter.strip()}\n", encoding="utf-8")
            resume_json_path.write_text(json.dumps(tailored_resume, indent=2), encoding="utf-8")

            metadata = {
                "base_resume_id": base_resume_id,
                "remote_resume_id": remote_resume_id,
                "generated_at": generated_at.isoformat(),
                "company": scored_job.job.company,
                "job_title": scored_job.job.title,
                "pdf_url": pdf_url,
                "pdf_path": str(pdf_path),
                "cover_letter_path": str(cover_letter_path),
                "resume_json_path": str(resume_json_path),
                "score_10": scored_job.score_10,
                "rationale": scored_job.rationale,
                "matched_skills": list(scored_job.matched_skills),
                "missing_signals": list(scored_job.missing_signals),
                "tailored_content": tailored_content.model_dump(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            return GeneratedApplicationArtifacts(
                base_resume_id=base_resume_id,
                remote_resume_id=remote_resume_id,
                company=scored_job.job.company,
                job_title=scored_job.job.title,
                generated_at=generated_at.isoformat(),
                artifact_dir=str(artifact_dir),
                pdf_url=pdf_url,
                pdf_path=str(pdf_path),
                cover_letter_path=str(cover_letter_path),
                resume_json_path=str(resume_json_path),
                metadata_path=str(metadata_path),
            )
        except Exception:
            raise
        finally:
            if remote_resume_id:
                try:
                    self.resume_deleter(rxresume_base_url, rxresume_api_key, remote_resume_id)
                except Exception:
                    # Cleanup is best-effort once local artifacts have been written.
                    pass
