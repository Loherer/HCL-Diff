from __future__ import annotations

import pandas as pd
from hcl_diff.data.manifests import audit_patient_isolation


def test_patient_isolation_accepts_disjoint_splits() -> None:
    train = pd.DataFrame({"case_id": ["a", "b"]})
    validation = pd.DataFrame({"case_id": ["c"]})
    test = pd.DataFrame({"case_id": ["d"]})
    report = audit_patient_isolation(train, validation, test)
    assert report["passed"] is True
    assert not any(report["overlaps"].values())


def test_patient_isolation_rejects_overlap() -> None:
    train = pd.DataFrame({"case_id": ["a", "b"]})
    validation = pd.DataFrame({"case_id": ["b"]})
    report = audit_patient_isolation(train, validation)
    assert report["passed"] is False
    assert report["overlaps"]["train_validation"] == ["b"]
