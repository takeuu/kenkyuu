import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from io import BytesIO
from pydub import AudioSegment
import tempfile
import scipy.signal as signal

# タイトル
st.title('Audio Feature Extractor App (Supports m4a)')

# 音声ファイルのアップロード
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # m4aファイルを一時的にwavに変換して保存するための処理
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        if uploaded_file.name.endswith('.m4a'):
            audio = AudioSegment.from_file(uploaded_file, format="m4a")
            audio.export(temp_file.name, format="wav")
            temp_file_path = temp_file.name
        else:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

    # 音声ファイルをlibrosaで読み込む
    y, sr = librosa.load(temp_file_path)
    st.write(f"Sample Rate: {sr}")

    # 特徴量の選択メニュー
    feature_choice = st.multiselect(
        "Select features to extract and visualize:",
        ["Waveform", "STFT", "Mel-Spectrogram", "Zero Crossing Rate", "Bandpass Filter", "Pitch Tracking", "Notch Filter"]
    )

    # 波形の可視化
    if "Waveform" in feature_choice:
        st.write("Waveform:")
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(y, sr=sr)
        st.pyplot(plt)

    # STFT（短時間フーリエ変換）
    if "STFT" in feature_choice:
        st.write("Short-Time Fourier Transform (STFT):")
        D = librosa.stft(y)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('STFT')
        st.pyplot(plt)

    # メルスペクトログラム
    if "Mel-Spectrogram" in feature_choice:
        st.write("Mel-Spectrogram:")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        st.pyplot(plt)

    # ゼロ交差率
    if "Zero Crossing Rate" in feature_choice:
        st.write("Zero Crossing Rate:")
        zcr = librosa.feature.zero_crossing_rate(y)
        plt.figure(figsize=(10, 6))
        plt.plot(zcr[0])
        plt.title("Zero Crossing Rate")
        plt.xlabel("Frame")
        plt.ylabel("Rate")
        st.pyplot(plt)

    # バンドパスフィルタ（500Hz～1000Hz）
    if "Bandpass Filter" in feature_choice:
        st.write("Bandpass Filtered Signal (500Hz - 1000Hz):")
        nyquist = 0.5 * sr
        low = 500 / nyquist
        high = 1000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.lfilter(b, a, y)
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(filtered_signal, sr=sr)
        plt.title("Bandpass Filtered Signal (500Hz - 1000Hz)")
        st.pyplot(plt)

    # ピッチ検出
    if "Pitch Tracking" in feature_choice:
        st.write("Pitch Tracking:")
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        plt.figure(figsize=(10, 6))
        plt.plot(pitch_values)
        plt.title("Pitch Tracking")
        plt.xlabel("Frame")
        plt.ylabel("Frequency (Hz)")
        st.pyplot(plt)

    # ノッチフィルタ
    if "Notch Filter" in feature_choice:
        st.write("Remove Specific Frequency (Notch Filter):")
        remove_freq = st.slider("Select frequency to remove (Hz)", 0, sr // 2, 60)
        q_factor = st.slider("Select Q-factor for notch filter", 0.1, 30.0, 10.0)
        notch_b, notch_a = signal.iirnotch(remove_freq / (0.5 * sr), q_factor)
        notch_filtered_signal = signal.lfilter(notch_b, notch_a, y)

        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(notch_filtered_signal, sr=sr)
        plt.title(f"Notch Filtered Signal (Removing {remove_freq} Hz)")
        st.pyplot(plt)

        # ノッチフィルタ後の音声ファイルを再生
        output_wav_notch = BytesIO()
        write(output_wav_notch, sr, (notch_filtered_signal * 32767).astype(np.int16))
        output_wav_notch.seek(0)
        st.audio(output_wav_notch, format='audio/wav')

        # ダウンロードリンク
        st.download_button("Download Notch Filtered Audio", output_wav_notch, file_name=f"notch_filtered_audio_{remove_freq}Hz.wav")
