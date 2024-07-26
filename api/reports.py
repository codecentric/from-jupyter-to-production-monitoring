from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
)
from evidently.metric_preset import TargetDriftPreset
from evidently.report import Report
import pandas as pd

from api.models import Applicant


def get_column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target="loan_status",
        prediction="loan_status",
        numerical_features=list(Applicant.get_type_fields(float)),
        categorical_features=list(Applicant.get_type_fields(str)),
    )


def build_model_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> str:
    model_performance_report = Report(
        metrics=[
            ClassificationPreset(),
        ]
    )
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=get_column_mapping(),
    )
    # TODO: to IO
    report_path = "reports/model_performance.html"
    model_performance_report.save_html(report_path)

    return report_path


def build_target_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> str:
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    report_path = "reports/target_drift.html"
    target_drift_report.save_html(report_path)

    return report_path
