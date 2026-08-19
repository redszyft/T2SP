from scipy import stats
from pydantic import BaseModel

class Noise(BaseModel):
    mean: float
    std: float

def find_noise(y):
    """Assume that residuals follow a normal distribution"""
    mean, std = stats.norm.fit(y)
    return y, Noise(mean=round(mean, 2), std=round(std, 2))