import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

from .conftest import add_scratch, make_photo

JOB = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def client(storage):
    with TestClient(app) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/models").json()
    assert body["providers"][-1] == "CPUExecutionProvider"
    assert set(body["models"]) == {"inpaint", "colorize", "superres"}
    assert all(not m["present"] for m in body["models"].values())


def test_process_happy_path(client, storage):
    img = add_scratch(make_photo(), (60, 20), (330, 270), 3)
    ok, buf = cv2.imencode(".png", img)
    (storage.uploads_dir / f"{JOB}.png").write_bytes(buf.tobytes())

    res = client.post("/process", json={"job_id": JOB, "input_name": f"{JOB}.png",
                                        "options": {"upscale": 2, "repair_sensitivity": 0.6}})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["result_name"] == "result.png" and body["width"] == 800 and body["height"] == 600
    assert [s["name"] for s in body["steps"]][0] == "load"
    out = cv2.imread(str(storage.results_dir / JOB / "result.png"))
    assert out.shape[:2] == (600, 800)
    assert client.get(f"/progress/{JOB}").status_code == 404  # progresso limpo ao final


@pytest.mark.parametrize("payload", [
    {"job_id": "../../etc", "input_name": "a.png"},
    {"job_id": JOB, "input_name": "../segredo.png"},
    {"job_id": JOB, "input_name": "a/b.png"},
    {"job_id": JOB, "input_name": "a.png", "options": {"upscale": 3}},
    {"job_id": JOB, "input_name": "a.png", "options": {"repair_sensitivity": 1.5}},
])
def test_process_rejects_unsafe_or_invalid_payloads(client, payload):
    assert client.post("/process", json=payload).status_code == 422


def test_process_missing_and_invalid_files(client, storage):
    assert client.post("/process", json={"job_id": JOB, "input_name": "sumiu.png"}).status_code == 404
    (storage.uploads_dir / "ruim.png").write_bytes(b"nada de imagem aqui")
    res = client.post("/process", json={"job_id": JOB, "input_name": "ruim.png"})
    assert res.status_code == 422 and "decodificar" in res.json()["detail"]


def test_progress_validates_job_id(client):
    assert client.get("/progress/nao-e-uuid").status_code == 422
