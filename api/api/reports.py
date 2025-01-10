from io import StringIO
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    TargetDriftPreset,
)
from evidently.report import Report
import pandas as pd

from api.models import Applicant


def get_column_mapping() -> ColumnMapping:
    # evidently requires a ColumnMapping for its report, to identify the
    # target column and data types
    return ColumnMapping(
        target="loan_status",
        prediction="loan_status",
        numerical_features=list(Applicant.get_type_fields(float)),
        categorical_features=list(Applicant.get_type_fields(str)),
    )


def build_model_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> StringIO:
    model_performance_report = Report(metrics=[ClassificationPreset()])
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=get_column_mapping(),
    )
    _file = StringIO()
    model_performance_report.save_html(_file)

    return _file


def build_target_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> StringIO:
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=get_column_mapping(),
    )
    _file = StringIO()
    target_drift_report.save_html(_file)

    return _file
