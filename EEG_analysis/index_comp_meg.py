import mne
import os
import matplotlib
from compute_spectral_exponent import compute_SpectralExponent
from genhurst import genhurst
import EntropyHub as EH
from functionEI import calculate_fei
import fathon
from fathon import fathonUtils as fu
from Avalanche import avalanche_pipeline
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from scipy.signal import welch, detrend


def dsearchn(x, value):
    dist_from_value = np.abs(x - value)
    return np.where(dist_from_value == dist_from_value.min())[0][0]

workpath = '/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar2/'
sub = os.listdir(workpath)

# spectral exponents
for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    SE = np.zeros([len(file), 256])

    for fid in range(len(file)):
        # ========= 1. 读取保存好的 Epochs 文件 =========
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        # epochs.plot()
        # ========= 2. 提取两类刺激条件的 epoch =========
        try:
            epochs_ec = epochs['eyec']
            # epochs_eo = epochs['eyeo']
        except:
            epochs_ec = epochs['LOFF']
            # epochs_eo = epochs['L-ON']

        # ========= 3. 计算功率谱密度（Welch 方法） =========
        # 返回的 psd 是 shape: (n_epochs, n_channels, n_freqs)

        # psd_ec, freqs = epochs_ec.compute_psd(method='welch', fmin=0.5, fmax=70).get_data(return_freqs=True)
        # psd_eo, _     = epochs_eo.compute_psd(method='welch', fmin=1, fmax=40).get_data(return_freqs=True)

        # mean_psd_ec = psd_ec.mean(axis=0)
        # mean_psd_eo = psd_eo.mean(axis=0)

        # ========= 4. 计算频谱均值指标=========
        # slope, inter, stats, vectors = compute_SpectralExponent(freqs, mean_psd_ec[150,:], do_plot=True)
        for elc in range(256):
            eeg_1chan = epochs_ec.get_data()
            fs = 1000  # use the sampling rate from your data (in Hz)
            frex, psd_1chan = welch(eeg_1chan[:, elc, :], fs=fs, nperseg=fs * 3, noverlap=fs * 2, detrend='linear')
            psd_1chan = psd_1chan.mean(axis=0)
            frband = [1, 40]
            frBins = np.arange(dsearchn(frex, frband[0]), dsearchn(frex, frband[1]) + 1)
            XX = frex[frBins]
            YY = psd_1chan[frBins]
            slope, inter, stats, vectors = compute_SpectralExponent(XX, YY, do_plot=False)
            SE[fid, elc] = slope

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    np.save(os.path.join(outpath, sub[sid], 'days.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'SE.npy'), SE[index,:])

    import matplotlib.pyplot as plt
    mean_SE = np.mean(SE, axis=1)
    plt.figure(figsize=(10, 6))
    index = np.argsort(days)
    days, mean_SE = days[index], mean_SE[index]
    plt.plot(days, mean_SE)
    plt.xlabel('Weeks')
    plt.ylabel('Averaged SE')
    plt.title('Changes')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(outpath, sub[sid], 'change_SE.png'))
    plt.close()

# fEI/DFA/Hurst
for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    fEI = np.zeros([len(file), 256])
    Hurst = np.zeros([len(file), 256])
    eponum = np.zeros(len(file))

    for fid in range(len(file)):
        # ========= 1. 读取保存好的 Epochs 文件 =========
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        # epochs.plot()
        # ========= 2. 提取两类刺激条件的 epoch =========
        try:
            epochs_ec = epochs['eyec'].get_data()
            # epochs_eo = epochs['eyeo'].get_data()
        except:
            epochs_ec = epochs['LOFF'].get_data()
            # epochs_eo = epochs['L-ON'].get_data()

        # ========= 4. 计算频谱均值指标=========
        eponum[fid] = epochs_ec.shape[0]
        EItemp = np.zeros([epochs_ec.shape[0],256])
        Htemp = np.zeros([epochs_ec.shape[0], 256])
        for epo in range(epochs_ec.shape[0]):
            EI, wAmp, wDNF = calculate_fei(np.abs(np.squeeze(epochs_ec[epo, :256, :]).T), 5000, 0.8)
            H = np.zeros(256)
            for ele in range(256):
                # H[ele] = genhurst(np.squeeze(epochs_ec[epo, ele, :].T), 2)
                signal = detrend(np.squeeze(epochs_ec[epo, ele, :].T))
                a = fu.toAggregated(signal)
                # initialize dfa object
                pydfa = fathon.DFA(a)
                # compute fluctuation function and Hurst exponent
                wins = fu.linRangeByStep(10, 2000, 100)
                n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=3)
                DFA, H_intercept = pydfa.fitFlucVec()
                H[ele] = DFA
                if H[ele]<0.6:
                    EI[ele] = np.nan
                    print('no LRTC electrode'+str(ele))

            EItemp[epo, :] = EI
            Htemp[epo, :] = H
        fEI[fid, :] = np.nanmean(EItemp, axis=0)
        Hurst[fid, :] = np.mean(Htemp, axis=0)

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    np.save(os.path.join(outpath, sub[sid], 'Weeks.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'fEI.npy'), fEI[index,:])
    np.save(os.path.join(outpath, sub[sid], 'DFA.npy'), Hurst[index, :])

# multiscale entropy
Mobj = EH.MSobject('DispEn')

for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    MSE = np.zeros([len(file), 256, 20])

    for fid in range(len(file)):
        # ========= 1. 读取保存好的 Epochs 文件 =========
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        # epochs.plot()
        # ========= 2. 提取两类刺激条件的 epoch =========
        try:
            epochs_ec = epochs['eyec']
            # epochs_eo = epochs['eyeo']
        except:
            epochs_ec = epochs['LOFF']
            # epochs_eo = epochs['L-ON']

        # ========= 3. 计算功率谱密度（Welch 方法） =========
        # 返回的 psd 是 shape: (n_epochs, n_channels, n_freqs)

        # psd_ec, freqs = epochs_ec.compute_psd(method='welch', fmin=0.5, fmax=70).get_data(return_freqs=True)
        # psd_eo, _     = epochs_eo.compute_psd(method='welch', fmin=1, fmax=40).get_data(return_freqs=True)

        # mean_psd_ec = psd_ec.mean(axis=0)
        # mean_psd_eo = psd_eo.mean(axis=0)

        # ========= 4. 计算频谱均值指标=========
        # slope, inter, stats, vectors = compute_SpectralExponent(freqs, mean_psd_ec[150,:], do_plot=True)
        eeg_1chan = epochs_ec.get_data()
        for elc in range(256):
            MSEtemp = np.zeros([eeg_1chan.shape[0], 20])
            for epo in range(eeg_1chan.shape[0]):
                MSEtemp[epo, :], CI = EH.MSEn(np.squeeze(eeg_1chan[epo, elc, :]), Mobj, Scales=20)
            MSE[fid, elc, :] = MSEtemp.mean(axis=0)

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    # np.save(os.path.join(outpath, sub[sid], 'days.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'MSE.npy'), MSE[index,:])
