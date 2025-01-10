from sqlmodel import Session

from api.models import ApplicantPrediction


def save_predictions(db_session: Session, record: ApplicantPrediction) -> None:
    """Save predictions to database.

    Args:
        predictions (pd.DataFrame): Pandas dataframe with predictions column.
    """

    db_session.add(record)
    db_session.commit()
