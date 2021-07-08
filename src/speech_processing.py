import librosa as lr
import preprocess as pp
import librosa.effects as lre
import librosa.feature as lrf
import numpy as np
SAMPLING_RATE = 44100
DCT_TYPE = 2
N_MFCC = 13

def get_data(CONFIG_FILE_PATH, FILES_PATH):
    data, labels = [], []
    with open(CONFIG_FILE_PATH, encoding = "utf8") as audio_signals:
        for line in audio_signals:
            sound_file = line.strip()
            data.append(extract_speech_features(FILES_PATH + sound_file))
            labels.append(sound_file[ : len(sound_file) - 4])
    return data, labels

def extract_speech_features(AUDIO_PATH):
    data, sr = lr.load(AUDIO_PATH, sr = SAMPLING_RATE)
    data = lre.preemphasis(data)
    data = pp.noise_gating(data)
    data, _ = lre.trim(data, top_db=40, frame_length=2048, hop_length=512)
    melspectogram = lrf.melspectrogram(y = data, sr = sr, window = "hamming")
    melspectogram = lr.amplitude_to_db(melspectogram)
    data = lrf.mfcc(y = data, sr = sr, S = melspectogram, n_mfcc = N_MFCC)
    data = np.delete(data, (0, 1, 2), axis = 0)
    delta = lrf.delta(data)
    delta2 = lrf.delta(data, order = 2)
    comprehensive_data = np.concatenate((data, delta, delta2)).T

    return comprehensive_data