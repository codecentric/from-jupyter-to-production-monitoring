import logging
import pandas as pd

from fastapi import Depends, FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse

import onnxruntime as rt
from sqlmodel import SQLModel, create_engine, Session

from api.load import load_current_data, load_reference_data
from api.models import Applicant, Prediction, ApplicantPrediction
from api.predictions import save_predictions
from api.reports import build_model_performance_report, build_target_drift_report


logging.basicConfig(
    level=logging.INFO, format="FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s"
)


db_url = "sqlite:///database.db"
engine = create_engine(db_url, echo=True)
SQLModel.metadata.create_all(engine)


app = FastAPI()
sess = rt.InferenceSession("models/loan_model.onnx")


def get_db_session():
    with Session(engine) as _session:
        yield _session


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse("<h1><i>Evidently + FastAPI</i></h1>")


@app.post("/predict")
def predict(
    applicant: Applicant,
    background_tasks: BackgroundTasks,
    db_session: Session = Depends(get_db_session),
) -> list[Prediction]:
    ins = applicant.to_onnx()
    label, probabilities = sess.run(None, ins)
    predicted = []
    for k, v in probabilities[0].items():
        prediction = Prediction(label=k, probability=v)
        predicted.append(prediction)
    record = ApplicantPrediction(**applicant.model_dump(), label=int(label[0]))

    # Save predictions to database (in the background)
    background_tasks.add_task(save_predictions, db_session, record)

    return predicted


@app.get("/monitor-model")
def monitor_model_performance(
    window_size: int = 3000, db_session: Session = Depends(get_db_session)
) -> FileResponse:
    logging.info("Read current data")
    current_data: pd.DataFrame = load_current_data(window_size, db_session)

    logging.info("Read reference data")
    reference_data = load_reference_data()

    logging.info("Build report")
    report_path: str = build_model_performance_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    logging.info("Return report as html")
    return FileResponse(report_path)


@app.get("/monitor-target")
def monitor_target_drift(window_size: int = 3000) -> FileResponse:
    logging.info("Read current data")
    current_data: pd.DataFrame = load_current_data(window_size)

    logging.info("Read reference data")
    reference_data = load_reference_data()

    logging.info("Build report")
    report_path: str = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    logging.info("Return report as html")
    return FileResponse(report_path)
