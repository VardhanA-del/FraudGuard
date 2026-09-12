import numpy as np
from fraudguard.monitoring.drift import population_stability_index
from fraudguard.monitoring.health import determine_system_health

def test_drift_and_health_status():
    reference_window = np.arange(100)
    assert population_stability_index(reference_window, reference_window) < 0.001
    assert determine_system_health(100, ["STABLE"]) == "STABLE"
    assert determine_system_health(100, ["DRIFT"]) == "DRIFT DETECTED"

