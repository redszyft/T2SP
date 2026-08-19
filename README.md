# T2SP
Implementation of the "Representing Time Series as Structured Programs for LLM Reasoning" paper (https://arxiv.org/html/2606.12481v1).

## Introduction
The motivation behind representing time series as structured programs is to allow LLMs to reason about temporal data without exploding the context window.

By decomposing a time series into well defined base components we can represent the series as a structure of those components that's human-readable and AI-friendly.

## Components
By applying a sequential decomposition method, we can separate the components that represent the original signal.

*   **Trend**: A smooth baseline of the data identified using **B-Spline** interpolation.
*   **Periodic**: Oscillations within the series identified as linear combination of sine waves with dominant frequencies.
*   **Events**: Localized anomalies or occurrences, modeled as **pikes** (for sudden changes) or **gaussian** events (for more gradual processes).
*   **Noise**: The residual stochastic component remaining after all structural elements are extracted (modelled as gaussian noise).

### How it Works
The algorithm follows a subtractive decomposition pipeline (`sp.py`):
1.  **Trend Removal**: The baseline trend is identified and subtracted from the raw signal.
2.  **Periodicity Extraction**: Sinusoidal components are identified within the detrended signal.
3.  **Event Detection**: Localized "events" (spikes/gaussians) are isolated from the deperiodized signal.
_Note: this part of the method received very little attention
in the paper, so the implenetation here is custom._
4.  **Noise Estimation**: The remaining signal is classified as noise.

### Configuration
The decomposition parameters `config.yaml`

