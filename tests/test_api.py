"""The HTTP API, driven with FastAPI's TestClient over fake services (no network, no CLIP)."""

import pytest
from fastapi.testclient import TestClient

from outfit.agents.llm import LLMError
from outfit.api.app import create_app
from outfit.api.errors import NotFound
from outfit.ingestion.review import EditError
from tests.factories import make_item
from tests.fakes import fake_services


def api(services) -> TestClient:
    return TestClient(create_app(services))


def test_health_reports_wardrobe_size_and_model():
    services = fake_services([make_item(), make_item()])
    with api(services) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "items": 2, "model": services.settings.agent_model}


def test_services_passed_in_are_used_and_not_closed():
    services = fake_services()
    with api(services) as client:
        assert client.app.state.services is services
    assert services.store.count() == 0  # the store is still open after shutdown


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NotFound("no item abc"), 404),
        (EditError("unknown field 'colour'"), 422),
        (LLMError("no usable reply"), 502),
    ],
)
def test_domain_errors_map_to_status_codes(error, status):
    app = create_app(fake_services())

    @app.get("/boom")
    def boom():
        raise error

    with TestClient(app) as client:
        response = client.get("/boom")
    assert response.status_code == status
    assert response.json() == {"detail": str(error)}


def test_openapi_docs_are_served():
    with api(fake_services()) as client:
        assert client.get("/docs").status_code == 200
        assert "/health" in client.get("/openapi.json").json()["paths"]
