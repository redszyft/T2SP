import numpy as np
from scipy.fft import fft, fftfreq
from pydantic import BaseModel


class Sinusoid(BaseModel):
    amplitude: float
    period: float

def find_top_freqs(y, N: int, top_k: int):
    xf = fftfreq(N)[:N//2]
    yf = fft(y)[:N//2]
    top_freqs_idx = np.abs(yf).argsort()[::-1][:top_k]
    return xf[top_freqs_idx]

def build_sine_matrix(x, top_freqs: list[float]):
    sine_matrix = []
    for freq in top_freqs:
        sine_matrix.extend([np.sin(2 * np.pi * freq * x), np.cos(2 * np.pi * freq * x)])
    return np.array(sine_matrix).T

def find_periodicity(x, y, top_k: int):
    N = len(x)
    top_freqs = find_top_freqs(y, N, top_k)

    sine_matrix = build_sine_matrix(x, top_freqs)

    # normal equation
    a = np.matmul(sine_matrix.T, sine_matrix)
    b = np.dot(sine_matrix.T, y)
    coeffs = np.dot(np.linalg.inv(a), b)

    y_recon = (coeffs * sine_matrix).sum(axis=1)

    sines = []
    for i, freq in enumerate(top_freqs):

        sin_amp = np.round(coeffs[2 * i], 2)
        cos_amp = np.round(coeffs[2 * i + 1], 2)
        period = np.round(1 / freq / N, 2)
        sines.extend([
            Sinusoid(amplitude=sin_amp, period=period),
            Sinusoid(amplitude=cos_amp, period=period)
        ])
    return y_recon, sines