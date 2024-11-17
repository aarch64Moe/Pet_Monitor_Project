
from scipy.signal import butter, cheby1, ellip, lfilter, wiener,filtfilt, medfilt

import scipy.signal as signal

# Bandpass filter
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, audio)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=0.7, highcut=2.5, fs=10, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)


def apply_median_filter(data, kernel_size=5):
    return medfilt(data, kernel_size)



# Function to apply Butterworth band-pass filter
def butterworth_bandpass_filter(data, lowcut=300.0, highcut=3000.0, fs=48000, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the data
    y = lfilter(b, a, data)
    return y


# Function to apply Chebyshev Type I band-pass filter
def chebyshev_bandpass_filter(data, lowcut=300.0, highcut=3000.0, fs=48000, order=5, ripple=0.5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Chebyshev Type I filter
    b, a = cheby1(order, ripple, [low, high], btype='band')

    # Apply the filter to the data
    y = lfilter(b, a, data)
    return y


# Function to apply Elliptic band-pass filter
def elliptic_bandpass_filter(data, lowcut=300.0, highcut=3000.0, fs=48000, order=5, ripple=0.5,
                             stopband_attenuation=40):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Elliptic filter
    b, a = ellip(order, ripple, stopband_attenuation, [low, high], btype='band')

    # Apply the filter to the data
    y = lfilter(b, a, data)
    return y


# Function to apply Wiener filter for noise reduction
def wiener_filter(data, mysize=29, noise=None):
    # Apply Wiener filter to the data
    y = wiener(data, mysize=mysize, noise=noise)
    return y


def high_pass_filter(new_value, prev_filtered_value, alpha=0.95):
    return alpha * (prev_filtered_value + new_value)


# Function to apply the chosen filter based on the filter_type variable
def apply_filter(data, filter_type, fs=48000):
    lowcut = 300.0
    highcut = 3000.0
    order = 5
    ripple = 0.5  # For Chebyshev and Elliptic
    stopband_attenuation = 40  # For Elliptic filter

    if filter_type == 'butterworth':
        return butterworth_bandpass_filter(data, lowcut, highcut, fs, order)
    elif filter_type == 'chebyshev':
        return chebyshev_bandpass_filter(data, lowcut, highcut, fs, order, ripple)
    elif filter_type == 'elliptic':
        return elliptic_bandpass_filter(data, lowcut, highcut, fs, order, ripple, stopband_attenuation)
    elif filter_type == 'wiener':
        return wiener_filter(data)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


