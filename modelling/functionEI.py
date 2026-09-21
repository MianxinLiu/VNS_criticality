import numpy as np
from scipy import signal

def calculate_fei(signal_in: np.ndarray,
                  window_size: int,
                  window_overlap: float):
    """
    Calculate functional Excitation-Inhibition balance (fEI).

    Parameters
    ----------
    signal_in : np.ndarray
        Amplitude envelope, shape (num_samples, num_channels).
    window_size : int
        Window length in samples.
    window_overlap : float
        Fractional overlap between successive windows (0-1).

    Returns
    -------
    EI   : np.ndarray, shape (num_channels,)
           fEI value per channel.
    wAmp : np.ndarray, shape (num_channels, num_windows)
           Mean amplitude in each window.
    wDNF : np.ndarray, shape (num_channels, num_windows)
           Detrended-normalized fluctuations per window.
    """
    length_signal, num_channels = signal_in.shape
    window_offset = int(np.floor(window_size * (1.0 - window_overlap)))

    # ----- helper: build index matrix (num_windows, window_size) -----
    window_starts = np.arange(0, length_signal - window_size + 1, window_offset)
    num_windows   = len(window_starts)
    idx = window_starts[:, None] + np.arange(window_size)  # broadcast

    # Pre-allocate outputs
    EI   = np.zeros(num_channels)
    wAmp = np.zeros((num_channels, num_windows))
    wDNF = np.zeros((num_channels, num_windows))

    for ch in range(num_channels):
        amp = signal_in[:, ch]                         # originalAmplitude
        profile = np.cumsum(amp - amp.mean())          # signalProfile (ii → iii)

        # --- Step iii: mean amplitude per window ---
        w_orig_amp = amp[idx].mean(axis=1)             # (num_windows,)
        x_amp = w_orig_amp[:, None]                    # broadcast to (num_windows,1)

        # --- Arrange signals into windows, normalize (iii → iv) ---
        x_signal = profile[idx] / x_amp                # (num_windows, window_size)
        x_signal = x_signal.T                          # shape (window_size, num_windows)

        # --- Detrend each window (iv → v) ---
        d_signal = signal.detrend(x_signal, axis=0, type='linear')

        # --- Std per window (v → vi) ---
        w_dnf = d_signal.std(axis=0, ddof=0)           # (num_windows,)

        # --- Correlation & fEI (vi → vii) ---
        if np.std(w_dnf) == 0 or np.std(w_orig_amp) == 0:
            r = 0.0   # avoid divide-by-zero
        else:
            r = np.corrcoef(w_dnf, w_orig_amp)[0, 1]
        EI[ch] = 1.0 - r

        # Save window-wise metrics
        wAmp[ch, :] = w_orig_amp
        wDNF[ch, :] = w_dnf

    return EI, wAmp, wDNF