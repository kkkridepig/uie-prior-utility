import torch
from uie_next.scientific_evaluation import prediction_utility_statistics


def test_gate_has_explicit_null_predicted_utility():
    result = prediction_utility_statistics({'alpha': torch.ones(1,1,2,2)}, {'U': torch.ones(1,1,2,2)})
    assert result['U_hat_mse'] is None
    assert result['U_hat_true_correlation'] is None
    assert result['U_hat_status'] == 'not_provided_by_method'
    assert result['U_positive_fraction'] == 1
