import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from io import BytesIO
import scipy.signal as signal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォントを指定

# タイトル
st.title('オーディオ特徴抽出アプリ (MP3/WAV)')

# サイドバーにメニューを追加
st.sidebar.title('メニュー')

# 音声ファイルのアップロード
st.sidebar.header("音声ファイルのアップロード")
uploaded_file1 = st.sidebar.file_uploader("音声ファイル1を選択してください", type=["wav", "mp3"])
uploaded_file2 = st.sidebar.file_uploader("音声ファイル2を選択してください", type=["wav", "mp3"])

# 音声の切り取り機能の追加
st.sidebar.header("音声の切り取り設定")
trim_option = st.sidebar.checkbox("音声を切り取る")
if trim_option:
    start_time = st.sidebar.number_input("開始時間（秒）", min_value=0.0, value=0.0, step=0.1)
    end_time = st.sidebar.number_input("終了時間（秒）", min_value=0.0, value=0.0, step=0.1)

# 特徴量の選択メニュー
feature_choice = st.sidebar.multiselect(
    "抽出して可視化する特徴を選択してください：",
    ["波形", "STFT", "メルスペクトログラム", "ゼロ交差率", "バンドパスフィルタ", "ピッチ検出", "ノッチフィルタ", "比較分析", "異常検知", "類似度評価", "高度な可視化", "統計情報"]
)

# 重ね合わせ表示のオプション
overlay_option = st.sidebar.checkbox("重ね合わせ表示（比較可能な場合）")

# アプリのメインロジック
def main():
    if uploaded_file1 is not None:
        # 音声ファイル1をlibrosaで読み込む
        y1, sr1 = librosa.load(uploaded_file1, sr=None)
        st.write(f"音声ファイル1のサンプリングレート: {sr1} Hz")
        # 音声の切り取り
        if trim_option:
            total_duration1 = librosa.get_duration(y=y1, sr=sr1)
            if end_time == 0.0 or end_time > total_duration1:
                end_time_adj = total_duration1
            else:
                end_time_adj = end_time
            if start_time >= end_time_adj:
                st.warning("開始時間は終了時間より小さくする必要があります。")
            else:
                start_sample1 = int(start_time * sr1)
                end_sample1 = int(end_time_adj * sr1)
                y1 = y1[start_sample1:end_sample1]
    else:
        y1, sr1 = None, None

    if uploaded_file2 is not None:
        # 音声ファイル2をlibrosaで読み込む
        y2, sr2 = librosa.load(uploaded_file2, sr=None)
        st.write(f"音声ファイル2のサンプリングレート: {sr2} Hz")
        # 音声の切り取り
        if trim_option:
            total_duration2 = librosa.get_duration(y=y2, sr=sr2)
            if end_time == 0.0 or end_time > total_duration2:
                end_time_adj = total_duration2
            else:
                end_time_adj = end_time
            if start_time >= end_time_adj:
                st.warning("開始時間は終了時間より小さくする必要があります。")
            else:
                start_sample2 = int(start_time * sr2)
                end_sample2 = int(end_time_adj * sr2)
                y2 = y2[start_sample2:end_sample2]
    else:
        y2, sr2 = None, None

    if y1 is not None or y2 is not None:
        # 選択された特徴量に応じて処理を実行
        if "波形" in feature_choice:
            plot_waveform(y1, sr1, y2, sr2)
        if "STFT" in feature_choice:
            plot_stft(y1, sr1, y2, sr2)
        if "メルスペクトログラム" in feature_choice:
            plot_melspectrogram(y1, sr1, y2, sr2)
        if "ゼロ交差率" in feature_choice:
            plot_zero_crossing_rate(y1, sr1, y2, sr2)
        if "バンドパスフィルタ" in feature_choice:
            plot_bandpass_filter(y1, sr1, y2, sr2)
        if "ピッチ検出" in feature_choice:
            plot_pitch(y1, sr1, y2, sr2)
        if "ノッチフィルタ" in feature_choice:
            plot_notch_filter(y1, sr1, y2, sr2)
        if "比較分析" in feature_choice:
            compare_analysis(y1, sr1, y2, sr2)
        if "類似度評価" in feature_choice:
            evaluate_similarity(y1, sr1, y2, sr2)
        if "異常検知" in feature_choice:
            detect_anomalies(y1, sr1, y2, sr2)
        if "高度な可視化" in feature_choice:
            advanced_visualization(y1, sr1, y2, sr2)
        if "統計情報" in feature_choice:
            show_statistics(y1, sr1, y2, sr2)
    else:
        st.write("音声ファイルをアップロードしてください。")

# 各機能の実装（関数化して整理）
def plot_waveform(y1, sr1, y2, sr2):
    st.write("波形：")
    if overlay_option and y1 is not None and y2 is not None:
        # 波形の重ね合わせ表示
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y1, sr=sr1, alpha=0.5, label='音声ファイル1')
        librosa.display.waveshow(y2, sr=sr2, color='r', alpha=0.5, label='音声ファイル2')
        plt.title("波形の重ね合わせ表示")
        plt.legend()
        st.pyplot(plt)
    else:
        if y1 is not None:
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y1, sr=sr1)
            plt.title("音声ファイル1の波形")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(y1, '音声ファイル1_波形.csv')
        if y2 is not None:
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y2, sr=sr2, color='r')
            plt.title("音声ファイル2の波形")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(y2, '音声ファイル2_波形.csv')

def plot_stft(y1, sr1, y2, sr2):
    st.write("短時間フーリエ変換（STFT）：")
    if overlay_option and y1 is not None and y2 is not None:
        st.write("重ね合わせ表示はSTFTには適用されません。個別に表示します。")
    if y1 is not None:
        D1 = librosa.stft(y1)
        S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_db1, sr=sr1, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('音声ファイル1のSTFT')
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(S_db1, '音声ファイル1_STFT.csv')
    if y2 is not None:
        D2 = librosa.stft(y2)
        S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_db2, sr=sr2, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('音声ファイル2のSTFT')
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(S_db2, '音声ファイル2_STFT.csv')

def plot_melspectrogram(y1, sr1, y2, sr2):
    st.write("メルスペクトログラム：")
    if overlay_option and y1 is not None and y2 is not None:
        st.write("重ね合わせ表示はメルスペクトログラムには適用されません。個別に表示します。")
    if y1 is not None:
        S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128, fmax=8000)
        S_dB1 = librosa.power_to_db(S1, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_dB1, sr=sr1, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('音声ファイル1のメルスペクトログラム')
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(S_dB1, '音声ファイル1_メルスペクトログラム.csv')
    if y2 is not None:
        S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128, fmax=8000)
        S_dB2 = librosa.power_to_db(S2, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(S_dB2, sr=sr2, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('音声ファイル2のメルスペクトログラム')
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(S_dB2, '音声ファイル2_メルスペクトログラム.csv')

def plot_zero_crossing_rate(y1, sr1, y2, sr2):
    st.write("ゼロ交差率：")
    if overlay_option and y1 is not None and y2 is not None:
        zcr1 = librosa.feature.zero_crossing_rate(y1)[0]
        zcr2 = librosa.feature.zero_crossing_rate(y2)[0]
        min_length = min(len(zcr1), len(zcr2))
        zcr1 = zcr1[:min_length]
        zcr2 = zcr2[:min_length]
        frames = range(len(zcr1))
        plt.figure(figsize=(10, 3))
        plt.plot(frames, zcr1, label='音声ファイル1')
        plt.plot(frames, zcr2, label='音声ファイル2', alpha=0.7)
        plt.title("ゼロ交差率の重ね合わせ表示")
        plt.xlabel("フレーム")
        plt.ylabel("レート")
        plt.legend()
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(pd.DataFrame({'音声ファイル1': zcr1, '音声ファイル2': zcr2}), 'ゼロ交差率_比較.csv')
    else:
        if y1 is not None:
            zcr1 = librosa.feature.zero_crossing_rate(y1)[0]
            plt.figure(figsize=(10, 3))
            plt.plot(zcr1)
            plt.title("音声ファイル1のゼロ交差率")
            plt.xlabel("フレーム")
            plt.ylabel("レート")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(zcr1, '音声ファイル1_ゼロ交差率.csv')
        if y2 is not None:
            zcr2 = librosa.feature.zero_crossing_rate(y2)[0]
            plt.figure(figsize=(10, 3))
            plt.plot(zcr2, color='r')
            plt.title("音声ファイル2のゼロ交差率")
            plt.xlabel("フレーム")
            plt.ylabel("レート")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(zcr2, '音声ファイル2_ゼロ交差率.csv')

def plot_bandpass_filter(y1, sr1, y2, sr2):
    st.write("バンドパスフィルタ適用（500Hz - 1000Hz）：")
    if y1 is not None:
        nyquist1 = 0.5 * sr1
        low = 500 / nyquist1
        high = 1000 / nyquist1
        b1, a1 = signal.butter(4, [low, high], btype='band')
        filtered_signal1 = signal.lfilter(b1, a1, y1)
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(filtered_signal1, sr=sr1)
        plt.title("音声ファイル1のバンドパスフィルタ適用信号")
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(filtered_signal1, '音声ファイル1_バンドパスフィルタ.csv')
    if y2 is not None:
        nyquist2 = 0.5 * sr2
        low = 500 / nyquist2
        high = 1000 / nyquist2
        b2, a2 = signal.butter(4, [low, high], btype='band')
        filtered_signal2 = signal.lfilter(b2, a2, y2)
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(filtered_signal2, sr=sr2, color='r')
        plt.title("音声ファイル2のバンドパスフィルタ適用信号")
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(filtered_signal2, '音声ファイル2_バンドパスフィルタ.csv')

def plot_pitch(y1, sr1, y2, sr2):
    st.write("ピッチ検出：")
    if overlay_option and y1 is not None and y2 is not None:
        pitch_values1 = extract_pitch(y1, sr1)
        pitch_values2 = extract_pitch(y2, sr2)
        min_length = min(len(pitch_values1), len(pitch_values2))
        pitch_values1 = pitch_values1[:min_length]
        pitch_values2 = pitch_values2[:min_length]
        frames = range(len(pitch_values1))
        plt.figure(figsize=(10, 3))
        plt.plot(frames, pitch_values1, label='音声ファイル1')
        plt.plot(frames, pitch_values2, label='音声ファイル2', alpha=0.7)
        plt.title("ピッチトラッキングの重ね合わせ表示")
        plt.xlabel("フレーム")
        plt.ylabel("周波数 (Hz)")
        plt.legend()
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(pd.DataFrame({'音声ファイル1': pitch_values1, '音声ファイル2': pitch_values2}), 'ピッチ検出_比較.csv')
    else:
        if y1 is not None:
            pitch_values1 = extract_pitch(y1, sr1)
            plt.figure(figsize=(10, 3))
            plt.plot(pitch_values1)
            plt.title("音声ファイル1のピッチトラッキング")
            plt.xlabel("フレーム")
            plt.ylabel("周波数 (Hz)")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(pitch_values1, '音声ファイル1_ピッチ検出.csv')
        if y2 is not None:
            pitch_values2 = extract_pitch(y2, sr2)
            plt.figure(figsize=(10, 3))
            plt.plot(pitch_values2, color='r')
            plt.title("音声ファイル2のピッチトラッキング")
            plt.xlabel("フレーム")
            plt.ylabel("周波数 (Hz)")
            st.pyplot(plt)
            # データをCSVに保存
            save_csv(pitch_values2, '音声ファイル2_ピッチ検出.csv')

def extract_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
        else:
            pitch_values.append(np.nan)
    return pitch_values

def plot_notch_filter(y1, sr1, y2, sr2):
    st.write("特定周波数の除去（ノッチフィルタ）：")
    max_sr = int(max(sr1 if y1 is not None else 0, sr2 if y2 is not None else 0))
    remove_freq = st.slider("除去する周波数を選択してください (Hz)", 0, max_sr // 2, 60)
    q_factor = st.slider("ノッチフィルタのQ値を選択してください", 0.1, 30.0, 10.0)
    if y1 is not None:
        notch_b1, notch_a1 = signal.iirnotch(remove_freq / (0.5 * sr1), q_factor)
        notch_filtered_signal1 = signal.lfilter(notch_b1, notch_a1, y1)
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(notch_filtered_signal1, sr=sr1)
        plt.title(f"音声ファイル1のノッチフィルタ適用信号（{remove_freq} Hzを除去）")
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(notch_filtered_signal1, f'音声ファイル1_ノッチフィルタ_{remove_freq}Hz.csv')
        # ノッチフィルタ後の音声ファイルを再生
        output_wav_notch1 = BytesIO()
        write(output_wav_notch1, sr1, (notch_filtered_signal1 * 32767).astype(np.int16))
        output_wav_notch1.seek(0)
        st.audio(output_wav_notch1, format='audio/wav')
        # ダウンロードリンク
        st.download_button(f"音声ファイル1のノッチフィルタ適用音声をダウンロード", output_wav_notch1, file_name=f"notch_filtered_audio1_{remove_freq}Hz.wav")
    if y2 is not None:
        notch_b2, notch_a2 = signal.iirnotch(remove_freq / (0.5 * sr2), q_factor)
        notch_filtered_signal2 = signal.lfilter(notch_b2, notch_a2, y2)
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(notch_filtered_signal2, sr=sr2, color='r')
        plt.title(f"音声ファイル2のノッチフィルタ適用信号（{remove_freq} Hzを除去）")
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(notch_filtered_signal2, f'音声ファイル2_ノッチフィルタ_{remove_freq}Hz.csv')
        # ノッチフィルタ後の音声ファイルを再生
        output_wav_notch2 = BytesIO()
        write(output_wav_notch2, sr2, (notch_filtered_signal2 * 32767).astype(np.int16))
        output_wav_notch2.seek(0)
        st.audio(output_wav_notch2, format='audio/wav')
        # ダウンロードリンク
        st.download_button(f"音声ファイル2のノッチフィルタ適用音声をダウンロード", output_wav_notch2, file_name=f"notch_filtered_audio2_{remove_freq}Hz.wav")

def compare_analysis(y1, sr1, y2, sr2):
    st.write("比較分析：")
    if y1 is not None and y2 is not None:
        # 波形の比較
        st.write("波形の比較：")
        plt.figure(figsize=(10, 6))
        plt.subplot(2,1,1)
        librosa.display.waveshow(y1, sr=sr1)
        plt.title("音声ファイル1の波形")
        plt.subplot(2,1,2)
        librosa.display.waveshow(y2, sr=sr2, color='r')
        plt.title("音声ファイル2の波形")
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(y1, '音声ファイル1_波形.csv')
        save_csv(y2, '音声ファイル2_波形.csv')
    else:
        st.write("比較分析には2つの音声ファイルが必要です。")

def evaluate_similarity(y1, sr1, y2, sr2):
    st.write("類似度評価：")
    if y1 is not None and y2 is not None:
        # MFCCを使用して類似度を計算
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20)
        # フレーム数を合わせる
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]
        # コサイン類似度を計算
        similarity = cosine_similarity(mfcc1.T, mfcc2.T)
        avg_similarity = np.mean(similarity)
        st.write(f"音声ファイル間の平均類似度（MFCCベース）：{avg_similarity:.4f}")
        # 類似度マトリックスをヒートマップで表示
        plt.figure(figsize=(6, 4))
        plt.imshow(similarity, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('MFCCベースの類似度マトリックス')
        plt.xlabel('音声ファイル2のフレーム')
        plt.ylabel('音声ファイル1のフレーム')
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(similarity, '類似度マトリックス.csv')
    else:
        st.write("類似度評価には2つの音声ファイルが必要です。")

def detect_anomalies(y1, sr1, y2, sr2):
    st.write("異常検知：")
    def anomaly_detection(y, sr, file_name):
        # エネルギーベースの異常検知（閾値を設定）
        rmse = librosa.feature.rms(y=y)[0]
        frames = range(len(rmse))
        times = librosa.frames_to_time(frames, sr=sr)
        threshold = np.mean(rmse) + 2 * np.std(rmse)
        anomalies = rmse > threshold
        plt.figure(figsize=(10, 3))
        plt.plot(times, rmse, label='エネルギー')
        plt.hlines(threshold, times[0], times[-1], colors='r', linestyles='dashed', label='閾値')
        plt.title(f"{file_name}のエネルギーと異常検知")
        plt.xlabel("時間 (秒)")
        plt.ylabel("エネルギー")
        plt.legend()
        st.pyplot(plt)
        # データをCSVに保存
        save_csv(pd.DataFrame({'時間': times, 'エネルギー': rmse, '異常': anomalies}), f'{file_name}_異常検知.csv')
    if y1 is not None:
        anomaly_detection(y1, sr1, "音声ファイル1")
    if y2 is not None:
        anomaly_detection(y2, sr2, "音声ファイル2")

def advanced_visualization(y1, sr1, y2, sr2):
    st.write("高度な可視化：")
    def plot_3d_spectrogram(y, sr, file_name):
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(S_db.shape[1]), np.arange(S_db.shape[0]))
        ax.plot_surface(X, Y, S_db, cmap='viridis')
        ax.set_xlabel('時間フレーム')
        ax.set_ylabel('周波数ビン')
        ax.set_zlabel('振幅 (dB)')
        plt.title(f"{file_name}の3Dスペクトログラム")
        st.pyplot(plt)
    if y1 is not None:
        plot_3d_spectrogram(y1, sr1, "音声ファイル1")
    if y2 is not None:
        plot_3d_spectrogram(y2, sr2, "音声ファイル2")

def show_statistics(y1, sr1, y2, sr2):
    st.write("統計情報：")
    def display_stats(y, sr, file_name):
        duration = librosa.get_duration(y=y, sr=sr)
        mean = np.mean(y)
        std = np.std(y)
        max_amp = np.max(y)
        min_amp = np.min(y)
        st.write(f"**{file_name}の統計情報**")
        st.write(f"- 長さ（秒）：{duration:.2f}")
        st.write(f"- 平均振幅：{mean:.4f}")
        st.write(f"- 振幅の標準偏差：{std:.4f}")
        st.write(f"- 最大振幅：{max_amp:.4f}")
        st.write(f"- 最小振幅：{min_amp:.4f}")
    if y1 is not None:
        display_stats(y1, sr1, "音声ファイル1")
    if y2 is not None:
        display_stats(y2, sr2, "音声ファイル2")

def save_csv(data, filename):
    csv = None
    if isinstance(data, np.ndarray):
        csv = pd.DataFrame(data).to_csv(index=False)
    elif isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
    elif isinstance(data, list):
        csv = pd.DataFrame(data).to_csv(index=False)
    else:
        st.write(f"{filename}の保存に失敗しました。データ形式：{type(data)}")
        return
    st.download_button(
        label=f"{filename} をダウンロード",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )

# メイン関数の呼び出し
if __name__ == '__main__':
    main()
