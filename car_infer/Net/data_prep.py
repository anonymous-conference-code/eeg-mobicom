import os
import numpy as np
import scipy.signal as signal
import torch
from sklearn.model_selection import train_test_split
from scipy.signal import cheb2ord
from sklearn.preprocessing import StandardScaler

class FilterBank:
    def __init__(self, fs, bands, pass_width=4, f_width=4):
        self.fs = fs
        self.f_trans = 2
        """
        4Hz ~ 40Hz
        self.f_pass: [ 4  8 12 16 20 24 28 32 36]
        """
        self.f_pass = np.arange(4, 40, pass_width)
        self.f_width = f_width
        self.gpass = bands[0]
        self.gstop = bands[1]
        self.filter_coeff = {}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs / 2  # 125 Hz

        for i, f_low_pass in enumerate(self.f_pass):
            """
            f_pass:
            [ 4 8]
            [ 8 12]
            [12 16]
            [16 20]
            [20 24]
            [24 28]
            [28 32]
            [32 36]
            [36 40]
            """
            f_pass = np.asarray([f_low_pass, f_low_pass + self.f_width])

            """
            f_stop:
            [ 2 10]
            [ 6 14]
            [10 18]
            [14 22]
            [18 26]
            [22 30]
            [26 34]
            [30 38]
            [34 42]
            """
            f_stop = np.asarray([f_pass[0] - self.f_trans, f_pass[1] + self.f_trans])

            wp = f_pass / Nyquist_freq
            ws = f_stop / Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i: {'b': b, 'a': a}})

        return self.filter_coeff

    def filter_data(self, eeg_data, window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape

        if window_details:
            n_samples = int(self.fs * (window_details.get('tmax') - window_details.get('tmin')))
            # +1

        filtered_data = np.zeros((len(self.filter_coeff), n_trials, n_channels, n_samples))

        for i, fb in self.filter_coeff.items():

            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b, a, eeg_data[j, :, :]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:, :, int((window_details.get('tmin')) * self.fs):int(
                    (window_details.get('tmax')) * self.fs)]
            filtered_data[i, :, :, :] = eeg_data_filtered

        return filtered_data

def _tensor_stack(x_fb):
    time_seg = [[0, 128*2], [128*1, 128*3]]
    # time_seg = [[0, 640]]
    temporal_seg = []
    for [a, b] in time_seg:
        temporal_seg.append(np.expand_dims(x_fb[:, :, :, a:b], axis=1))
    temporal_seg = np.concatenate(temporal_seg, axis=1)
    stack_tensor = temporal_seg

    return stack_tensor


def get_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T).T
    data = data[np.newaxis, :, :]
    print(data.shape)
    fbank = FilterBank(fs=128, bands=[3, 30], pass_width=4)
    _ = fbank.get_filter_coeff()
    X_train = _tensor_stack(fbank.filter_data(data).transpose(1, 0, 2, 3))
    return X_train

if __name__ == "__main__":
    data = torch.randn(16, 128*3)
    a = get_data(data)
    print(a.shape)




