import math
from Diffusion import calculate_time_step

GRID_SPACING = 1.0
DIFFUSIVITY = 1.0
TIME_STEP = 0.5
TOLERANCE = 0.01


def test_time_step():
    time_step = calculate_time_step(GRID_SPACING, DIFFUSIVITY)
    assert type(time_step) is float
    assert time_step == 0.5
    assert math.isclose(time_step, TIME_STEP, rel_tol=TOLERANCE)