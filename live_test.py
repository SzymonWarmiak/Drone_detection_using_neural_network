# Author: Szymon Warmiak

import numpy as np
import json
import threading
import queue
import sys
import time

try:
    import sounddevice as sd
except ImportError:
    print("Brakuje biblioteki: sounddevice")
    print("Uruchom: pip install sounddevice")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Brakuje biblioteki: matplotlib")
    print("Uruchom: pip install matplotlib")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("Brakuje biblioteki: librosa")
    print("Uruchom: pip install librosa")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TFLITE_BACKEND = "tensorflow"
    except ImportError:
        print("Brakuje TFLite! Uruchom: pip install tflite-runtime lub pip install tensorflow")
        sys.exit(1)

SAMPLE_RATE     = 16000
FRAME_SIZE      = 1024
HOP_LENGTH      = 1024
N_MELS          = 20
FMIN            = 80.0
FMAX            = 8000.0
NUM_FRAMES      = 16
WINDOW          = "hann"

MODEL_PATH      = "model_output/drone_model.tflite"
NORM_PATH       = "model_output/norm_params.json"

AUDIO_BUFFER_SAMPLES = FRAME_SIZE * NUM_FRAMES  # 16384 próbek ~ 1.024 s przy 16kHz
DETECTION_THRESHOLD  = 0.5

with open(NORM_PATH, "r") as f:
    norm = json.load(f)
NORM_MEAN = norm["mean"]
NORM_STD  = norm["std"]

if TFLITE_BACKEND == "tflite_runtime":
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
else:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"[OK] Model zaladowany ({TFLITE_BACKEND})")
print(f"   Wejscie: {input_details[0]['shape']} typ={input_details[0]['dtype'].__name__}")
print(f"   Wyjscie: {output_details[0]['shape']}")

audio_buffer  = np.zeros(AUDIO_BUFFER_SAMPLES, dtype=np.float32)
audio_lock    = threading.Lock()
result_queue  = queue.Queue(maxsize=5)

def extract_mel_spectrogram(audio_chunk: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=SAMPLE_RATE,
        n_fft=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        window=WINDOW,
        center=False
    )
    log_mel = np.log10(mel + 1e-10)              # [n_mels, n_frames]
    log_mel = log_mel.T                           # [n_frames, n_mels]
    log_mel = (log_mel - NORM_MEAN) / NORM_STD   # normalizacja Z-score
    return log_mel.astype(np.float32)            # [16, 20]

def run_inference(mel_spectrogram: np.ndarray) -> float:
    inp = mel_spectrogram[np.newaxis, :, :, np.newaxis]  # [1, 16, 20, 1]
    inp_detail = input_details[0]

    scale, zero_point = inp_detail['quantization']
    if scale == 0:
        scale = 1.0
    inp_q = np.clip(np.round(inp / scale + zero_point), -128, 127).astype(np.int8)

    interpreter.set_tensor(inp_detail['index'], inp_q)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    out_detail = output_details[0]
    out_scale, out_zp = out_detail['quantization']
    if out_scale == 0:
        out_scale = 1.0
    confidence = float((output.flatten()[0].astype(np.float32) - out_zp) * out_scale)
    return float(np.clip(confidence, 0.0, 1.0))

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    if status:
        pass  # np. overflow - ignorujemy
    mono = indata[:, 0].astype(np.float32)
    with audio_lock:
        audio_buffer = np.roll(audio_buffer, -len(mono))
        audio_buffer[-len(mono):] = mono

detection_running = True

def detection_thread_func():
    while detection_running:
        with audio_lock:
            chunk = audio_buffer.copy()

        mel = extract_mel_spectrogram(chunk)
        confidence = run_inference(mel)

        window_fn = np.hanning(FRAME_SIZE)
        recent_frame = chunk[-FRAME_SIZE:]
        fft_mag = np.abs(np.fft.rfft(recent_frame * window_fn))
        freqs = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / SAMPLE_RATE)

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
        result_queue.put({
            "confidence": confidence,
            "mel": mel,
            "fft_mag": fft_mag,
            "freqs": freqs,
            "audio": chunk[-2048:]  # ostatnie 2048 próbek do waveformu
        })
        time.sleep(0.1)  # 10 aktualizacji/s

plt.style.use("dark_background")

fig = plt.figure(figsize=(14, 9), facecolor="#0d0d0d")
fig.suptitle("Drone Detector - Live AI Monitor", fontsize=16,
             color="white", fontweight="bold", y=0.97)

gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
              top=0.92, bottom=0.07, left=0.08, right=0.96)

ax_wave  = fig.add_subplot(gs[0, :])   # Amplituda (pełna szerokość)
ax_fft   = fig.add_subplot(gs[1, 0])  # FFT
ax_mel   = fig.add_subplot(gs[1, 1])  # Mel-Spectrogram
ax_gauge = fig.add_subplot(gs[2, :])  # Wskaźnik pewności (pełna szerokość)

wave_x = np.linspace(0, 2048 / SAMPLE_RATE * 1000, 2048)
line_wave, = ax_wave.plot(wave_x, np.zeros(2048), color="#00e5ff", lw=0.8)
ax_wave.set_facecolor("#111111")
ax_wave.set_xlim(0, wave_x[-1])
ax_wave.set_ylim(-1.0, 1.0)
ax_wave.set_title("Amplituda dźwięku", color="#aaaaaa", fontsize=10, pad=4)
ax_wave.set_xlabel("Czas [ms]", color="#777777", fontsize=8)
ax_wave.set_ylabel("Amplituda", color="#777777", fontsize=8)
ax_wave.tick_params(colors="#555555", labelsize=7)
ax_wave.spines[:].set_color("#333333")

fft_bins = FRAME_SIZE // 2 + 1
freq_axis = np.fft.rfftfreq(FRAME_SIZE, d=1.0 / SAMPLE_RATE)
line_fft, = ax_fft.plot(freq_axis / 1000, np.zeros(fft_bins),
                         color="#ff6f00", lw=0.9)
ax_fft.set_facecolor("#111111")
ax_fft.set_xlim(0, SAMPLE_RATE / 2 / 1000)
ax_fft.set_ylim(0, 1)
ax_fft.set_title("Widmo FFT", color="#aaaaaa", fontsize=10, pad=4)
ax_fft.set_xlabel("Częstotliwość [kHz]", color="#777777", fontsize=8)
ax_fft.set_ylabel("Magnituda (norm.)", color="#777777", fontsize=8)
ax_fft.tick_params(colors="#555555", labelsize=7)
ax_fft.spines[:].set_color("#333333")
ax_fft.axvspan(0, FMIN / 1000, alpha=0.15, color="gray")
ax_fft.axvspan(FMAX / 1000, SAMPLE_RATE / 2 / 1000, alpha=0.15, color="gray")

mel_img = ax_mel.imshow(np.zeros((N_MELS, NUM_FRAMES)), aspect="auto",
                         origin="lower", cmap="inferno",
                         vmin=-3, vmax=3,
                         extent=[0, NUM_FRAMES, 0, N_MELS])
ax_mel.set_facecolor("#111111")
ax_mel.set_title("Mel-Spectrogram (wejście AI)", color="#aaaaaa", fontsize=10, pad=4)
ax_mel.set_xlabel("Ramki", color="#777777", fontsize=8)
ax_mel.set_ylabel("Pasma Mel", color="#777777", fontsize=8)
ax_mel.tick_params(colors="#555555", labelsize=7)
plt.colorbar(mel_img, ax=ax_mel, label="Z-score", pad=0.02)

ax_gauge.set_facecolor("#111111")
ax_gauge.set_xlim(0, 1)
ax_gauge.set_ylim(0, 1)
ax_gauge.set_title("Pewność detekcji drona", color="#aaaaaa", fontsize=10, pad=4)
ax_gauge.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_gauge.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                           color="#777777", fontsize=9)
ax_gauge.set_yticks([])
ax_gauge.spines[:].set_color("#333333")

for x in np.linspace(0, 1, 200):
    color = (min(1, 2 * x), min(1, 2 * (1 - x)), 0.1)
    ax_gauge.axvspan(x, x + 0.005, alpha=0.25, color=color, linewidth=0)

bar_gauge = ax_gauge.barh(0.5, 0, height=0.55, color="#00e676", alpha=0.9, zorder=5)[0]
line_thresh = ax_gauge.axvline(DETECTION_THRESHOLD, color="white",
                                lw=2, ls="--", alpha=0.7, zorder=6)
ax_gauge.text(DETECTION_THRESHOLD + 0.01, 0.85, f"Próg ({int(DETECTION_THRESHOLD*100)}%)",
              color="white", fontsize=8, va="top")

text_result  = ax_gauge.text(0.5, 0.5, "Inicjalizacja...",
                               color="white", fontsize=22, fontweight="bold",
                               ha="center", va="center", zorder=10,
                               transform=ax_gauge.transAxes)
text_percent = ax_gauge.text(0.5, 0.12, "",
                               color="#cccccc", fontsize=11,
                               ha="center", va="bottom", zorder=10,
                               transform=ax_gauge.transAxes)

_last_confidence = [0.0]

def update(frame):
    try:
        data = result_queue.get_nowait()
    except queue.Empty:
        return

    conf     = data["confidence"]
    mel      = data["mel"]       # [16, 20]
    fft_mag  = data["fft_mag"]
    audio    = data["audio"]

    n = min(len(audio), 2048)
    wave_data = np.zeros(2048)
    wave_data[-n:] = audio[-n:]
    line_wave.set_ydata(wave_data)
    peak = max(np.max(np.abs(wave_data)) * 1.1, 0.01)
    ax_wave.set_ylim(-peak, peak)

    fft_norm = fft_mag / (np.max(fft_mag) + 1e-10)
    line_fft.set_ydata(fft_norm)

    mel_img.set_data(mel.T)  # transponujemy: [n_mels, n_frames]

    smooth_conf = 0.7 * _last_confidence[0] + 0.3 * conf
    _last_confidence[0] = smooth_conf

    bar_gauge.set_width(smooth_conf)
    r = min(1.0, 2.0 * (1.0 - smooth_conf))
    g = min(1.0, 2.0 * smooth_conf)
    bar_gauge.set_color((r, g, 0.1))

    if smooth_conf >= DETECTION_THRESHOLD:
        text_result.set_text(">>> DRON WYKRYTY! <<<")
        text_result.set_color("#ff1744")
    else:
        text_result.set_text("OK - Brak drona")
        text_result.set_color("#00e676")

    text_percent.set_text(f"Pewność: {smooth_conf * 100:.1f}%")

ani = animation.FuncAnimation(fig, update, interval=100, cache_frame_data=False)

if __name__ == "__main__":
    print("\nDostepne urzadzenia audio:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            marker = " <- (domyslne)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']} (wejscia: {dev['max_input_channels']}){marker}")

    device_id = None
    if len(sys.argv) > 1:
        try:
            device_id = int(sys.argv[1])
            print(f"\n[MIC] Uzywam urzadzenia [{device_id}]: {devices[device_id]['name']}")
        except (ValueError, IndexError):
            print("❌ Nieprawidłowe ID urządzenia, używam domyślnego.")

    if device_id is None:
        print(f"\n[MIC] Uzywam domyslnego urzadzenia wejsciowego.")
        print("   Mozesz podac ID urzadzenia jako argument: python live_test.py 2\n")

    det_thread = threading.Thread(target=detection_thread_func, daemon=True)
    det_thread.start()

    stream = sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=512,
        dtype="float32",
        callback=audio_callback,
    )

    try:
        with stream:
            print("[MIC] Mikrofon aktywny. Zamknij okno, aby zakonczyc.\n")
            plt.show()
    except KeyboardInterrupt:
        print("\n[STOP] Zatrzymano.")
    finally:
        detection_running = False
