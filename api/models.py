import numpy as np
from sqlmodel import Field, SQLModel


class Applicant(SQLModel):
    applicantincome: float
    coapplicantincome: float
    loanamount: float
    loan_amount_term: float
    credit_history: float
    married: str
    dependents: str
    education: str
    self_employed: str
    property_area: str

    @classmethod
    def get_type_fields(cls, _type) -> set:
        return {
            field
            for field, fieldinfo in cls.model_fields.items()
            if fieldinfo.annotation == _type
        }

    def to_onnx(self) -> dict:
        ins = {k: np.array(v) for k, v in self.model_dump().items()}
        for c in self.get_type_fields(float):
            ins[c] = ins[c].astype(np.float32).reshape((1, 1))
        for k in self.get_type_fields(str):
            ins[k] = ins[k].astype(object).reshape((1, 1)).astype(object)

        return ins


class Prediction(SQLModel):
    label: int
    probability: float


class ApplicantPrediction(Applicant, table=True):
    id: int | None = Field(default=None, primary_key=True)
    label: int
