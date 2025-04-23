import logging
from typing import Annotated

import onnxruntime as rt
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import HTMLResponse
from sqlmodel import Session, SQLModel, create_engine

from api.load import load_current_data, load_reference_data
from api.models import Applicant, ApplicantPrediction, Prediction
from api.predictions import save_predictions
from api.reports import build_model_performance_report, build_target_drift_report

logging.basicConfig(
    level=logging.INFO, format="FASTAPI_APP - %(asctime)s - %(levelname)s - %(message)s"
)


# setup sqlite database
db_url = "sqlite:///database.db"
engine = create_engine(db_url, echo=True)
SQLModel.metadata.create_all(engine)


# setup api and load model
app = FastAPI()
sess = rt.InferenceSession("/models/loan_model.onnx")


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
    db_session: Annotated[Session, Depends(get_db_session)],
) -> list[Prediction]:
    # convert request data to onnx compatible format
    ins = applicant.to_onnx()
    # predict
    label, probabilities = sess.run(None, ins)

    # convert prediction into response and database compatible format
    predicted = []
    for k, v in probabilities[0].items():
        prediction = Prediction(label=k, probability=v)
        predicted.append(prediction)

    # Save predictions to database (in the background)
    record = ApplicantPrediction(**applicant.model_dump(), label=int(label[0]))
    background_tasks.add_task(save_predictions, db_session, record)

    return predicted


@app.get("/monitor-model")
def monitor_model_performance(
    db_session: Annotated[Session, Depends(get_db_session)],
    window_size: int = 3000,
) -> HTMLResponse:
    logging.info("Read current data")
    current_data: pd.DataFrame = load_current_data(window_size, db_session)

    if current_data.empty:
        return HTMLResponse("No data to monitor")

    logging.info("Read reference data")
    reference_data = load_reference_data()

    logging.info("Build report")
    report = build_model_performance_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    logging.info("Return report as html")
    return HTMLResponse(report.getvalue())


@app.get("/monitor-target")
def monitor_target_drift(
    db_session: Annotated[Session, Depends(get_db_session)],
    window_size: int = 3000,
) -> HTMLResponse:
    logging.info("Read current data")
    current_data: pd.DataFrame = load_current_data(window_size, db_session)

    if current_data.empty:
        return HTMLResponse("No data to monitor")

    logging.info("Read reference data")
    reference_data = load_reference_data()

    logging.info("Build report")
    report = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    logging.info("Return report as html")
    return HTMLResponse(report.getvalue())
