from fastapi.testclient import TestClient

from avalanche_api.app import create_app

ANSWER = '{"problems": [{"type": "gliding_snow", "danger_level": 1}]}'


class FakeModel:
    """Returns a fixed answer and records what it was asked."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.texts: list[str] = []

    def generate(self, text: str) -> str:
        self.texts.append(text)
        return self.answer


def client(model: FakeModel) -> TestClient:
    return TestClient(create_app(load=lambda: model))


def test_extract_returns_the_parsed_bulletin():
    model = FakeModel(ANSWER)
    with client(model) as c:
        response = c.post("/extract", json={"text": "Gliding avalanches are possible."})

    assert response.status_code == 200
    assert response.json()["problems"][0]["type"] == "gliding_snow"
    assert response.json()["problems"][0]["aspects"] == []
    assert model.texts == ["Gliding avalanches are possible."]


def test_invalid_model_answer_is_a_502_with_the_raw_output():
    with client(FakeModel('{"problems": [{"type": "avalanche"}]}')) as c:
        response = c.post("/extract", json={"text": "Gliding avalanches are possible."})

    assert response.status_code == 502
    assert response.json()["output"] == '{"problems": [{"type": "avalanche"}]}'


def test_empty_text_is_rejected_before_the_model_runs():
    model = FakeModel(ANSWER)
    with client(model) as c:
        response = c.post("/extract", json={"text": ""})

    assert response.status_code == 422
    assert model.texts == []


def test_root_redirects_to_swagger_ui():
    with client(FakeModel(ANSWER)) as c:
        response = c.get("/", follow_redirects=False)
        docs = c.get("/docs")

    assert response.headers["location"] == "/docs"
    assert docs.status_code == 200


def test_swagger_ui_offers_the_example_bulletin():
    with client(FakeModel(ANSWER)) as c:
        schema = c.get("/openapi.json").json()["components"]["schemas"]["ExtractRequest"]

    assert "bise wind" in schema["examples"][0]["text"]
