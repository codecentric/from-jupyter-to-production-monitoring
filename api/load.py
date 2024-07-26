import pandas as pd

from sqlmodel import Session, select

from api.models import ApplicantPrediction


def load_current_data(window_size: int, db_session: Session) -> pd.DataFrame:
    with db_session:
        # TODO: order_by / created_at
        records = db_session.exec(
            select(ApplicantPrediction)
            .order_by(ApplicantPrediction.id.desc())
            .limit(window_size)
        )

        current_data: pd.DataFrame = pd.DataFrame.from_records(
            [record.model_dump(exclude={"id"}) for record in records]
        )
        current_data.rename(columns={"label": "loan_status"}, inplace=True)
    return current_data


def load_reference_data() -> pd.DataFrame:
    ref_path = "data/reference_data.csv"
    ref_data = pd.read_csv(ref_path)
    return ref_data
