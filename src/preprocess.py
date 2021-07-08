import numpy as np

def noise_gating(time_series):
    min_value, max_value = np.min(time_series), np.max(time_series)
    PERCENTAGE = 0.08
    threshold_value = PERCENTAGE * (abs(max_value) - abs(min_value))
    threshold = (-threshold_value, threshold_value)

    for i in range(len(time_series)):
        if time_series[i] >= threshold[0] and time_series[i] <= threshold[1]:
            time_series[i] = 0
    return time_series