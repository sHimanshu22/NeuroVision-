"""Optional, backend-only Gemini explanations for structured model results."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from google.genai.errors import APIError as GoogleAPIError
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    GoogleAPIError = ()

try:
    import httpx
except ImportError:  # pragma: no cover - dependency is installed with google-genai
    httpx = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
logger = logging.getLogger(__name__)


class ClassificationAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    class_probabilities: dict[str, float] = Field(default_factory=dict)


class SegmentationAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "Not performed"
    original_width: int | None = Field(default=None, ge=1)
    original_height: int | None = Field(default=None, ge=1)
    processed_size: int | None = Field(default=None, ge=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    foreground_pixels: int | None = Field(default=None, ge=0)
    sparse_warning: str | None = None


class GradcamAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class GeminiAnalysisRequest(BaseModel):
    """Allowlisted fields sent by the browser; images and identifiers are rejected."""

    model_config = ConfigDict(extra="forbid")

    classification: ClassificationAnalysis | None = None
    segmentation: SegmentationAnalysis | None = None
    gradcam: GradcamAnalysis | None = None


class GeminiExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plain_language_summary: str = Field(min_length=1)
    classification_explanation: str = Field(min_length=1)
    segmentation_explanation: str = Field(min_length=1)
    gradcam_explanation: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    next_steps: str = Field(min_length=1)
    educational_terms: list[str] = Field(min_length=1)


class GeminiServiceError(Exception):
    """Safe error intended for user-facing handling."""


class GeminiNotConfiguredError(GeminiServiceError):
    pass


class GeminiInvalidResponseError(GeminiServiceError):
    pass


class GeminiProviderError(GeminiServiceError):
    pass


def _status_category(status_code: object) -> str:
    if status_code in (401, 403):
        return "authentication"
    if status_code == 404:
        return "model_not_found"
    if status_code == 400:
        return "invalid_request"
    if status_code == 429:
        return "quota_rate_limit"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "provider_server"
    return "unknown"


def _is_google_api_error(error: BaseException) -> bool:
    return bool(GoogleAPIError) and isinstance(error, GoogleAPIError)


def _log_provider_failure(error: BaseException, *, category: str | None = None) -> None:
    status_code = getattr(error, "code", None) if _is_google_api_error(error) else None
    safe_category = category or _status_category(status_code)
    safe_status = status_code if isinstance(status_code, int) else "unknown"
    logger.error(
        "Gemini request failed: category=%s status=%s",
        safe_category,
        safe_status,
    )


def _raise_safe_provider_error(error: BaseException) -> None:
    if _is_google_api_error(error):
        _log_provider_failure(error)
    elif isinstance(error, (TimeoutError, ConnectionError)) or (
        httpx is not None
        and isinstance(error, (httpx.TimeoutException, httpx.NetworkError))
    ):
        _log_provider_failure(error, category="network_timeout")
    else:
        _log_provider_failure(error, category="unknown")
    raise GeminiProviderError("Gemini explanations are temporarily unavailable.") from None


def configured_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _get_client(api_key: str):
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - covered by deployment dependency
        raise GeminiNotConfiguredError("Gemini explanations are unavailable on this server.") from exc

    try:
        return genai.Client(api_key=api_key)
    except GoogleAPIError as exc:
        _raise_safe_provider_error(exc)
    except Exception as exc:
        _raise_safe_provider_error(exc)


def _prompt(analysis: GeminiAnalysisRequest) -> str:
    structured_results = json.dumps(
        analysis.model_dump(exclude_none=True),
        ensure_ascii=True,
        sort_keys=True,
    )
    return f"""
You explain NeuroVision research-prototype outputs to a patient or family member
who may not have a technical background.
Use only the structured JSON below. Do not invent, infer, or alter any result.
If a section is null or absent, say that it was not performed.

Rules:
- Never diagnose, recommend treatment, or claim a person is healthy or disease-free.
- A predicted class is a computer-generated category, not a confirmed diagnosis.
- A label such as "pituitary" is only the AI model's predicted category. It does not confirm a pituitary tumor or any disease. Apply the same careful wording to other labels.
- Confidence is model confidence, not the probability that a person has a disease.
- When confidence is available, state it as a percentage to exactly two decimal places.
- Segmentation pixels are model-selected foreground pixels; an empty or sparse mask is
  inconclusive and never proves that a tumor is absent.
- Grad-CAM highlights regions that influenced classification and does not prove a tumor
  location or that the prediction is correct.
- Mention that NeuroVision is an educational research prototype and not a clinical tool.
- Use short, calm, everyday sentences. Avoid unexplained technical terms.
- Fill the response fields as follows:
  - plain_language_summary: a short overview of the available results.
  - "What the AI found" (classification_explanation): explain the category and how to
    understand its confidence.
  - "What the segmentation means" (segmentation_explanation): explain the mask, including
    that a sparse or empty mask
    is inconclusive rather than proof that a tumor is absent.
  - "What the highlighted regions mean" (gradcam_explanation): explain what highlighted
    regions mean and what they do not prove.
  - limitations: explain the main limits of these model outputs.
  - "What to do next" (next_steps): advise consulting a qualified doctor or radiologist who can review the
    full MRI, official radiology report, symptoms, and medical history. State that medical decisions must not be based on NeuroVision alone.
  - educational_terms: short explanations of any necessary terms.
- Return only the requested JSON schema.

Structured results:
{structured_results}
""".strip()


def _validate_provider_response(response: Any) -> GeminiExplanation:
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiInvalidResponseError("Gemini returned an invalid explanation.")

    try:
        return GeminiExplanation.model_validate_json(text)
    except (TypeError, ValueError):
        raise GeminiInvalidResponseError("Gemini returned an invalid explanation.") from None


def generate_gemini_explanation(
    analysis: GeminiAnalysisRequest,
    *,
    client: Any | None = None,
) -> GeminiExplanation:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiNotConfiguredError("Gemini explanations are not enabled on this server.")

    active_client = client or _get_client(api_key)
    model = configured_model()

    try:
        # This verifies that the configured project can access the exact model. There
        # is deliberately no fallback to another model or to a paid tier.
        active_client.models.get(model=model)
        response = active_client.models.generate_content(
            model=model,
            contents=_prompt(analysis),
            config={
                "response_mime_type": "application/json",
                "response_json_schema": GeminiExplanation.model_json_schema(),
            },
        )
    except GeminiServiceError:
        raise
    except GoogleAPIError as exc:
        _raise_safe_provider_error(exc)
    except Exception as exc:
        _raise_safe_provider_error(exc)

    return _validate_provider_response(response)
