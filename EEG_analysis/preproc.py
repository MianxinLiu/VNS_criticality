import mne
import os
import glob
import matplotlib
import numpy as np
from autoreject import AutoReject
from mne_icalabel import label_components

workpath = '/mnt/pci-0000:00:17.0-ata-6/VNS/VNS-EEG/2nd_wave/'
sub = os.listdir(workpath)
ar = AutoReject(cv=3)

for sid in range(9, len(sub)):
    mfffile = os.listdir(os.path.join(workpath, sub[sid]))
    for fid in range(len(mfffile)-1):
        # ========= 1. 读取 .mff 数据 =========
        # 注意：这里传入的是 .mff 文件夹路径，而不是单个文件
        path = glob.glob(os.path.join(workpath, sub[sid], mfffile[fid])+'/*mff')
        raw = mne.io.read_raw_egi(path[0], preload=True)
        # raw = mne.io.read_raw_egi(os.path.join(workpath, sub[sid], mfffile[fid]), preload=True)

        # ========= 2. 查看通道信息（可选） =========
        print(raw.info)
        print(raw.ch_names)

        # ========= 3. 设置 EEG 参考（如平均参考）=========
        # raw.plot()
        # raw.set_eeg_reference(ref_channels=['VREF'])
        raw.drop_channels(['VREF'])
        raw.set_eeg_reference('average', projection=True)

        # ========= 4. 滤波处理（去除低频漂移/高频噪声）=========
        # raw.resample(250)
        raw.filter(l_freq=.5, h_freq=80.)
        raw.notch_filter(50)
        # ========= 5. ICA 去除眼动/肌电等伪迹 =========
        try:
            ica = mne.preprocessing.ICA(n_components=0.99, method='fastica', random_state=1, max_iter=1000)
            ica.fit(raw)

            labels = label_components(raw, ica, 'iclabel')
            print(labels['labels'])  # 如 'brain', 'eye', 'muscle', ...
            ica.exclude = [i for i, l in enumerate(labels['labels']) if l not in ('other', 'brain')]

            # 应用 ICA 去伪迹
            raw_clean = ica.apply(raw.copy())
            if len(labels['labels']) == len(ica.exclude):
                raw_clean = raw.copy()
        except:
            print('only one IC')
            raw_clean = raw.copy()


        # ========= 6. 查找事件（trigger channel，可能是 'STI 014' 或 'TRIGGER'）=========
        # 你可以先用 raw.ch_names 看一下事件通道名称
        ecpos = np.where(np.array(raw.ch_names) == 'eyec')[0]
        eopos = np.where(np.array(raw.ch_names) == 'eyeo')[0]

        if ecpos.size == 0:
            ecpos = np.where(np.array(raw.ch_names) == 'LOFF')[0]
            eopos = np.where(np.array(raw.ch_names) == 'L-ON')[0]

        if ecpos.size != 0:
            print('ecpos    ', ecpos[0], 'eopos    ', eopos[0])
            events = mne.find_events(raw_clean, stim_channel=[raw.ch_names[ecpos[0]], raw.ch_names[eopos[0]]], shortest_event=1)
            # print(events[:5])  # 查看前几个事件
            # ========= 7. 设置事件类型（根据你的 paradigm）=========
            event_id = {raw.ch_names[ecpos[0]]: ecpos[0]-255, raw.ch_names[eopos[0]]: eopos[0]-255}  # 根据实际 trigger 编号修改
        else:
            events = np.load("./backupevents.npy")
            event_id = {'eyec': 2, 'eyeo': 1}

        # ========= 8. 切成 epochs =========
        epochs = mne.Epochs(
            raw_clean, events=events, event_id=event_id,
            tmin=2, tmax=87,  # 每段从前0.1s到后90s
            baseline=(2.0, 2.2),
            preload=True
        )

        # ========= 9. 自动剔除电压过大的 trial（伪迹）=========
        # epochs.drop_bad(reject=dict(eeg=150e-6))  # 超过150μV的试次会被剔除
        epochs, reject_log = ar.fit_transform(epochs, return_log=True)

        # ========= 10. 可视化与保存 =========
        # epochs.plot()  # 可选：可视化每个trial
        os.makedirs(os.path.join('/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar2/', sub[sid]), exist_ok=True)
        fname = mfffile[fid]
        # fname = mfffile[fid].split('.mf')[0]
        # fname = fname.split('-')[1]
        savepath = os.path.join('/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar2/', sub[sid], fname)

        epochs.save(savepath+'.fif', overwrite=True)

        # ========= 11. 平均生成 ERP（如果需要）=========
        # evoked = epochs['stimulus/left'].average()
        # evoked.plot(title="ERP for stimulus/left")