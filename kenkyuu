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
st.title('Audio Frequency Extractor App (Supports m4a)')

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

    # 短時間フーリエ変換（STFT）
    D = librosa.stft(y)

    # 周波数軸と時間軸の設定
    frequencies = librosa.fft_frequencies(sr=sr)

    # 周波数帯域の範囲を選択
    freq_min = st.slider("Select minimum frequency (Hz)", 0, int(frequencies.max()), 500)
    freq_max = st.slider("Select maximum frequency (Hz)", freq_min, int(frequencies.max()), 1000)

    # 特定の周波数帯域を抽出
    freq_indices = np.where((frequencies >= freq_min) & (frequencies <= freq_max))[0]
    filtered_D = np.zeros_like(D)
    filtered_D[freq_indices, :] = D[freq_indices, :]

    # 逆STFTで音声信号に戻す
    filtered_y = librosa.istft(filtered_D)

    # 音声ファイルとして保存（メモリ上に）
    output_wav = BytesIO()
    write(output_wav, sr, (filtered_y * 32767).astype(np.int16))
    output_wav.seek(0)

    # 波形の可視化
    st.write("Filtered Frequency Signal:")
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(filtered_y, sr=sr)
    st.pyplot(plt)

    # 音声ファイルの再生
    st.audio(output_wav, format='audio/wav')

    # ダウンロードリンク
    st.download_button("Download Filtered Audio", output_wav, file_name="filtered_audio.wav")

    # メルスペクトログラムの生成と表示
    st.write("Mel-Spectrogram:")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    st.pyplot(plt)

    # ゼロ交差率（Zero Crossing Rate）の計算と表示
    st.write("Zero Crossing Rate:")
    zcr = librosa.feature.zero_crossing_rate(y)
    plt.figure(figsize=(10, 6))
    plt.plot(zcr[0])
    plt.title("Zero Crossing Rate")
    plt.xlabel("Frame")
    plt.ylabel("Rate")
    st.pyplot(plt)

    # バンドパスフィルタリング（500Hz～1000Hz）
    st.write("Bandpass Filtered Signal (500Hz - 1000Hz):")
    nyquist = 0.5 * sr
    low = 500 / nyquist
    high = 1000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.lfilter(b, a, y)

    # フィルタ後の信号を波形として表示
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(filtered_signal, sr=sr)
    plt.title("Bandpass Filtered Signal (500Hz - 1000Hz)")
    st.pyplot(plt)

    # ピッチ検出と表示
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
