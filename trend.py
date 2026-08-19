import numpy as np
from scipy.interpolate import make_lsq_spline
from pydantic import BaseModel

class BSplineTrend(BaseModel):
    degree: int 
    n_knots: int 
    knots: list[float] 
    knots_ts_vals: list[float]


def find_trend(x, y, n_knots: int, degree: int):
    internal_knots = np.linspace(0, len(x), n_knots)[1:-1]
    knots = np.concatenate([
        [x[0]] * (degree + 1),
        internal_knots,
        [x[-1]] * (degree + 1)
    ])

    spline = make_lsq_spline(x, y, t=knots)
    unique_knots = sorted(list(set(knots)))
    knots_ts_vals = np.round(spline(unique_knots), 2)

    return spline(x), BSplineTrend(
        degree=spline.k,
        n_knots=n_knots,
        knots=unique_knots,
        knots_ts_vals=knots_ts_vals
    )