import numpy as np
from scipy.optimize import curve_fit
from scipy.signal.windows import gaussian
from pydantic import BaseModel

class SpikeEvent(BaseModel):
    center: int
    amplitude: float

class GaussianEvent(BaseModel):
    center: int
    amplitude: float
    width: float

def fit_gaussian(x, std):
    return gaussian(len(x), std=std) 

def peak_width(y, event_time, cut_off):
    """Find the bounding x values for the peak width at cutoff value"""

    below = np.argwhere(y < cut_off).flatten()
    left_candidate = below[below < event_time][-1]
    right_candidate = below[below > event_time][0]
    step = max(event_time - left_candidate, right_candidate - event_time)
    return event_time - step, event_time + step

def fit_peak(y, event_time: int):
    cut_off = y[event_time] / 2
    left_bound, right_bound = peak_width(y, event_time, cut_off)

    amp = np.round(y[event_time], 2)
    if right_bound - left_bound == 2:
        return left_bound, right_bound, SpikeEvent(center=event_time, amplitude=amp)
                
    peak = y[left_bound:(right_bound)+1]  
    norm_peak = np.abs(peak)/np.abs(peak).max()
    std, _ = curve_fit(fit_gaussian, list(range(len(peak))), norm_peak)
    return left_bound, right_bound, GaussianEvent(
        center=event_time,
        amplitude=amp,
        width=np.round(std, 2)
    )

def reconstruct_spike(y, center, left_bound, right_bound):
    return np.array([
        y[left_bound], y[center], y[right_bound]
    ])

def reconstruct_gaussian(left_bound, right_bound, gauss: GaussianEvent):
    size = right_bound - left_bound + 1
    return  gauss.amplitude * gaussian(size, std=gauss.width)

def find_events(y, percentile):
    y_abs = np.abs(y)
    event_threshold = np.percentile(y_abs, percentile)
    event_times = np.argwhere(y_abs > event_threshold).flatten()

    events = []
    y_recon = np.zeros(len(y))
    for event_time in event_times:
        left_bound, right_bound, fitted_event = fit_peak(y, event_time)
        if isinstance(fitted_event, SpikeEvent):
            y_recon[left_bound:right_bound+1] = reconstruct_spike(y, fitted_event.center, left_bound, right_bound)
        else:
            y_recon[left_bound:right_bound+1] = reconstruct_gaussian(left_bound, right_bound, fitted_event)
        events.append(fitted_event)

    return y_recon, events
