from sp import Series

model_desc = """
## Trend
represents the smooth baseline trend over time
- degree: the degree of the B-spline (e.g., 1 for linear, 3 for cubic. Should always be >= 1)
- knots: define where the trend is allowed to change its shape (change points in time)
- n_knots: The number of knots where the trend is allowed to change its shape
- knot_ts_value: the time-series value at each knot. This is the actual value (containing the noise).

## Periodic
represents periodic oscillations within the series
- amplitude: strength of oscillation
- period: length of one cycle

## Events
we account for two type of events: spikes and gaussian events

### GaussianEvent
represents localized transient events
- center: where the event occurs
- width: how spread the event is over time and is present only for gaussian events
- amplitude: magnitude of the event

### SpikeEvent
represents instantaneous sharp anomalies
- center: time of occurrence
- amplitude: magnitude of the spike

## Residual
represents unexplained noise
- mean: noise center
- std: noise intensity
"""

def prompt_example(data: Series):
    prompt_template = f"""
    You are a time-series expert.
    Your task is to analyze the symbolic program that represents a time series and provide an accurate description of the underlying data.

    # Description of the structured program representation
    The structured program representation is composed of the below interpretable components:
    
    {model_desc}

    The final time series is obtained by combining all components additively.
    Each component has a clear semantic meaning, and the structured program representation provides a structured, interpretable representation of the signal.
    
    Here's the structured data representing the time series:
    {data.model_dump()}
    """

    return prompt_template