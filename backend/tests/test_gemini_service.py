import json
import logging

import pytest
from google.genai.errors import APIError

from backend.app import gemini_service


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requested_model = None
        self.generate_kwargs = None

    def get(self, *, model):
        self.requested_model = model
        if self.error:
            raise self.error
        return object()

    def generate_content(self, **kwargs):
        self.generate_kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


def valid_response():
    return type("Response", (), {"text": json.dumps({
        "plain_language_summary": "A plain-language summary.",
        "classification_explanation": "The class is a model category.",
        "segmentation_explanation": "The mask shows selected pixels.",
        "gradcam_explanation": "The heatmap shows influential regions.",
        "limitations": "This is not a clinical diagnosis.",
        "next_steps": "Discuss the full MRI and radiology report with a qualified doctor or radiologist.",
        "educational_terms": ["model confidence"],
    })})()


def complete_analysis():
    return gemini_service.GeminiAnalysisRequest.model_validate({
        "classification": {
            "predicted_class": "glioma",
            "confidence": 0.8,
            "class_probabilities": {"glioma": 0.8, "normal": 0.2},
        },
        "segmentation": {
            "status": "Completed",
            "processed_size": 256,
            "threshold": 0.25,
            "foreground_pixels": 100,
        },
        "gradcam": {"predicted_class": "glioma", "confidence": 0.8},
    })


def test_success_uses_configured_model_and_validates_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    client = FakeClient(response=valid_response())

    result = gemini_service.generate_gemini_explanation(complete_analysis(), client=client)

    assert result.plain_language_summary == "A plain-language summary."
    assert client.models.requested_model == "gemini-3.5-flash-lite"
    config = client.models.generate_kwargs["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == gemini_service.GeminiExplanation.model_json_schema()
    assert "response_schema" not in config


def test_missing_key_fails_without_calling_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(gemini_service.GeminiNotConfiguredError):
        gemini_service.generate_gemini_explanation(complete_analysis())


def test_invalid_provider_response_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = FakeClient(response=type("Response", (), {"text": "{}"})())

    with pytest.raises(gemini_service.GeminiInvalidResponseError):
        gemini_service.generate_gemini_explanation(complete_analysis(), client=client)


def test_provider_failure_is_safe(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = FakeClient(error=RuntimeError("provider failure with sensitive details"))

    with pytest.raises(gemini_service.GeminiProviderError) as error:
        gemini_service.generate_gemini_explanation(complete_analysis(), client=client)

    assert "sensitive" not in str(error.value)


@pytest.mark.parametrize(
    ("status_code", "category"),
    [
        (401, "authentication"),
        (403, "authentication"),
        (404, "model_not_found"),
        (400, "invalid_request"),
        (429, "quota_rate_limit"),
        (500, "provider_server"),
        (503, "provider_server"),
    ],
)
def test_google_api_statuses_are_categorized(status_code, category):
    assert gemini_service._status_category(status_code) == category


def test_api_error_logs_only_safe_category_and_status(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    secret_text = "secret response body with api-key=do-not-log"
    provider_error = APIError(
        401,
        {"error": {"status": "UNAUTHENTICATED", "message": secret_text}},
    )
    client = FakeClient(error=provider_error)

    with caplog.at_level(logging.ERROR, logger="backend.app.gemini_service"):
        with pytest.raises(gemini_service.GeminiProviderError) as error:
            gemini_service.generate_gemini_explanation(complete_analysis(), client=client)

    assert str(error.value) == "Gemini explanations are temporarily unavailable."
    assert "category=authentication status=401" in caplog.text
    assert secret_text not in caplog.text
    assert "api-key" not in caplog.text


def test_unexpected_error_logs_unknown_without_exposing_details(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    secret_text = "unexpected secret header Authorization=Bearer do-not-log"
    client = FakeClient(error=RuntimeError(secret_text))

    with caplog.at_level(logging.ERROR, logger="backend.app.gemini_service"):
        with pytest.raises(gemini_service.GeminiProviderError):
            gemini_service.generate_gemini_explanation(complete_analysis(), client=client)

    assert "category=unknown status=unknown" in caplog.text
    assert secret_text not in caplog.text
    assert "Authorization" not in caplog.text


def test_incomplete_analysis_is_allowed_and_missing_sections_are_preserved():
    request = gemini_service.GeminiAnalysisRequest.model_validate({
        "classification": None,
        "segmentation": None,
        "gradcam": {"predicted_class": "normal", "confidence": 0.6},
    })

    assert request.classification is None
    assert request.segmentation is None
    assert request.gradcam.predicted_class == "normal"


def test_unallowlisted_image_or_identifier_fields_are_rejected():
    with pytest.raises(ValueError):
        gemini_service.GeminiAnalysisRequest.model_validate({
            "classification": {"predicted_class": "normal"},
            "filename": "scan.jpg",
        })


def test_prompt_requires_patient_friendly_sections_and_safe_confidence_wording():
    prompt = gemini_service._prompt(complete_analysis())

    assert "What the AI found" in prompt
    assert 'A label such as "pituitary" is only the AI model\'s predicted category.' in prompt
    assert "does not confirm a pituitary tumor or any disease" in prompt
    assert "percentage to exactly two decimal places" in prompt
    assert "not the probability that a person has a disease" in prompt
    assert "qualified doctor or radiologist" in prompt
    assert "medical decisions must not be based on NeuroVision alone" in prompt


def test_prompt_explains_sparse_segmentation_without_claiming_absence():
    prompt = gemini_service._prompt(complete_analysis())

    assert "sparse or empty mask" in prompt
    assert "never proves that a tumor is absent" in prompt


def test_prompt_marks_missing_analyses_as_not_performed():
    request = gemini_service.GeminiAnalysisRequest.model_validate({
        "classification": None,
        "segmentation": None,
        "gradcam": None,
    })

    prompt = gemini_service._prompt(request)

    assert "Structured results:\n{}" in prompt
    assert "If a section is null or absent, say that it was not performed." in prompt
