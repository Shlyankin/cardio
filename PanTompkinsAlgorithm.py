import scipy.signal as signal
import numpy as np

def pan_tompkins_detect(unfiltered_ecg, fs):
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm.
        In: IEEE Transactions on Biomedical Engineering
        BME-32.3 (1985), pp. 230–236.
        """

        f1 = 5 / fs
        f2 = 15 / fs

        b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = np.diff(filtered_ecg)

        squared = diff * diff

        N = int(0.12 * fs)
        mwa = MWA(squared, N)
        mwa[:int(0.12 * fs)] = 0
        mwa_peaks = panPeakDetect(mwa, fs)
        N = int(0.4 * fs)
        mwa = MWA(squared, N)
        mwa[:int(0.12 * fs)] = 0
        starts, ends = find_intervals(mwa, mwa_peaks)

        annot = np.zeros(len(unfiltered_ecg), np.int64)
        for i in range(len(starts)):
            annot[starts[i] : ends[i]] = 1
        return annot

def find_intervals(signal, pks):
    pks = np.insert(pks, 0, 0)
    pks = np.append(pks, len(signal) - 1)
    starts = []
    ends = []
    for i in range(1, len(pks) - 1):
        interval = range(int((pks[i - 1] + pks[i]) / 2), int((pks[i] + pks[i+1]) / 2))
        trheshold = signal[interval].max() * 0.2
        flag = False
        for j in interval:
            if not flag and signal[j] > trheshold:
                starts.append(j)
                flag = True
            if flag and signal[j] < trheshold:
                ends.append(j)
                flag = False
                break
        if flag:
            starts.pop(len(starts) - 1)
    return np.array(starts), np.array(ends)

def MWA(input_array, window_size):
        mwa = np.zeros(len(input_array))
        for i in range(len(input_array)):
                if i < window_size:
                        section = input_array[0:i]
                else:
                        section = input_array[i - window_size:i]

                if i != 0:
                        mwa[i] = np.mean(section)
                else:
                        mwa[i] = input_array[i]

        return mwa


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks
