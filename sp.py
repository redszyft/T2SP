import yaml
from trend import find_trend, BSplineTrend
from periodic import find_periodicity, Sinusoid
from events import find_events, SpikeEvent, GaussianEvent
from noise import find_noise, Noise
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Sequence

class Series(BaseModel):
    trend: BSplineTrend = Field(description='represents the smooth baseline trend of the series over time')
    periodic: list[Sinusoid] = Field(description='represents periodic oscillations within the series')
    events: list[SpikeEvent | GaussianEvent] = Field(description='represents localized events in the series')
    noise: Noise = Field(description='represents unexplained noise in the series')

BASE_DIR = Path(__file__).parent
with open( BASE_DIR / 'config.yaml') as cfg_file:
    config = yaml.safe_load(cfg_file)

def t2sp(x: Sequence[int], y: Sequence[float]):
    trend_recon, trend_comp = find_trend(x, y, **config['trend'])
    detrended = y - trend_recon

    periodic_recon, periodic_comp = find_periodicity(x, detrended, **config['periodic'])
    deperiodized = detrended - periodic_recon

    events_recon, events_comp = find_events(deperiodized, **config['events'])
    deevented = deperiodized - events_recon

    noise_recon, noise_comp = find_noise(deevented)

    return Series(
        trend=trend_comp,
        periodic=periodic_comp,
        events=events_comp,
        noise=noise_comp
    )
