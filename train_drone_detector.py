# Author: Szymon Warmiak
#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import json
import glob
import time
import pathlib
import datetime
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

SAMPLE_RATE     = 16_000       # Hz  (TIM3 trigger frequency)
FRAME_SIZE      = 1024         # samples per FFT  (= DMA half-buffer)
HOP_LENGTH      = 1024         # no overlap  (Ping-Pong HT/TC)
POWER           = 2.0

N_MELS          = 20           # number of Mel bins  (compact, drone BPF ≈ 80–4 kHz)
FMIN            = 80.0         # Hz — capture rotor fundamentals
FMAX            = 8000.0       # Hz — Nyquist at 16 kHz

NUM_FRAMES      = 16           # temporal context ≈ 16 × 1024/16000 = 1.024 s
WINDOW_SAMPLES  = NUM_FRAMES * HOP_LENGTH   # 16 384 samples per input clip

EPOCHS          = 80
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
DROPOUT_CONV    = 0.25         # after each conv block
DROPOUT_DENSE   = 0.50         # before output layer
RANDOM_SEED     = 42

SCRIPT_DIR      = pathlib.Path(__file__).resolve().parent
DATASET_ROOT    = SCRIPT_DIR / "Binary_Drone_Audio"
OUTPUT_DIR      = SCRIPT_DIR / "model_output"

CLASS_MAP       = {"yes_drone": 1, "unknown": 0}    # label encoding

MAX_ARENA_KB    = 25           # tensor-arena ceiling (KB)

def scan_dataset() -> tuple[list[str], list[int]]:
    paths, labels = [], []
    sr_set: set[int] = set()

    for class_name, label in CLASS_MAP.items():
        class_dir = DATASET_ROOT / class_name
        if not class_dir.is_dir():
            sys.exit(f"[ERROR] Missing class directory: {class_dir}")
        wavs = sorted(glob.glob(str(class_dir / "*.wav")))
        if not wavs:
            sys.exit(f"[ERROR] No .wav files found in {class_dir}")
        for wf in wavs[:5]:
            info = librosa.get_samplerate(wf)
            sr_set.add(info)
        paths.extend(wavs)
        labels.extend([label] * len(wavs))
        print(f"  {class_name:>12s}: {len(wavs):>6d} files")

    print(f"  Detected sample-rates: {sr_set}")
    print(f"  Total files: {len(paths)}")
    return paths, labels

def extract_mfe(audio: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        win_length=FRAME_SIZE,
        window="hann",
        center=False,          # CRITICAL – MCU processes block-by-block
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=POWER,           # power spectrum  ↔  arm_cmplx_mag_squared_f32
    )
    S_log = np.log10(S + 1e-10)
    return S_log.T             # → (time_frames, n_mels)

def load_and_extract(path: str) -> np.ndarray | None:
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [WARN] skipping {path}: {e}")
        return None

    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)), mode="constant")

    windows = []
    for start in range(0, len(y) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        chunk = y[start : start + WINDOW_SAMPLES]
        mfe = extract_mfe(chunk)               # (NUM_FRAMES, N_MELS)
        if mfe.shape[0] >= NUM_FRAMES:
            windows.append(mfe[:NUM_FRAMES])
    return np.array(windows) if windows else None

def build_dataset(paths: list[str], labels: list[int]):
    X_all, y_all = [], []
    total = len(paths)

    for i, (path, label) in enumerate(zip(paths, labels)):
        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"    [{i+1:>5d}/{total}]  processing ...")
        feats = load_and_extract(path)
        if feats is not None:
            X_all.append(feats)
            y_all.extend([label] * len(feats))

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.array(y_all, dtype=np.float32)
    return X, y

def compute_norm_stats(X: np.ndarray) -> tuple[float, float]:
    mu  = float(np.mean(X))
    std = float(np.std(X))
    return mu, std

def apply_norm(X: np.ndarray, mu: float, std: float) -> np.ndarray:
    return ((X - mu) / (std + 1e-10)).astype(np.float32)

def build_model(input_shape: tuple[int, ...]) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="mfe_input")

    x = keras.layers.Conv2D(8, (3, 3), padding="valid", activation="relu",
                            name="conv1")(inp)
    x = keras.layers.MaxPool2D((2, 2), name="pool1")(x)
    x = keras.layers.Dropout(DROPOUT_CONV, name="drop1")(x)

    x = keras.layers.Conv2D(16, (3, 3), padding="valid", activation="relu",
                            name="conv2")(x)
    x = keras.layers.MaxPool2D((2, 2), name="pool2")(x)
    x = keras.layers.Dropout(DROPOUT_CONV, name="drop2")(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(16, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(DROPOUT_DENSE, name="drop3")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="drone_detector")
    return model

def train(model: keras.Model,
          X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray,
          class_weight: dict | None = None) -> keras.callbacks.History:

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    return history

def evaluate(model, X_test, y_test, tag="Float32"):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_true = y_test.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\n{'═' * 50}")
    print(f"  {tag} Model Evaluation")
    print(f"{'═' * 50}")
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"       Pred 0  Pred 1")
    print(f"  T 0  {tn:>5d}   {fp:>5d}")
    print(f"  T 1  {fn:>5d}   {tp:>5d}")
    print(f"{'═' * 50}\n")
    return {"loss": loss, "acc": acc, "precision": precision,
            "recall": recall, "f1": f1}

def quantise_int8(model: keras.Model,
                  X_cal: np.ndarray,
                  output_path: pathlib.Path) -> bytes:
    def representative_dataset():
        indices = np.random.default_rng(RANDOM_SEED).choice(
            len(X_cal), size=min(200, len(X_cal)), replace=False
        )
        for i in indices:
            sample = X_cal[i : i + 1]
            yield [sample.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"  Saved TFLite model → {output_path}")
    print(f"  Model size: {len(tflite_model):,d} bytes  "
          f"({len(tflite_model)/1024:.1f} KB)")
    return tflite_model

def verify_tflite(tflite_bytes: bytes,
                  X_test: np.ndarray, y_test: np.ndarray,
                  norm_mu: float, norm_std: float):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    inp_scale, inp_zp = inp_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]

    arena_bytes = 0
    for td in interpreter.get_tensor_details():
        shape = td["shape"]
        arena_bytes += int(np.prod(shape))   # INT8 → 1 byte per element
    print(f"\n  Estimated tensor arena ≈ {arena_bytes:,d} bytes "
          f"({arena_bytes/1024:.1f} KB)")
    if arena_bytes / 1024 > MAX_ARENA_KB:
        print(f"  [WARNING] Arena exceeds {MAX_ARENA_KB} KB ceiling!")
    else:
        print(f"  [OK] Arena fits within {MAX_ARENA_KB} KB ceiling.")

    correct = 0
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        q_input = np.round(sample / inp_scale + inp_zp).astype(np.int8)
        interpreter.set_tensor(inp_detail["index"], q_input)
        interpreter.invoke()
        q_output = interpreter.get_tensor(out_detail["index"])
        output_float = (q_output.astype(np.float32) - out_zp) * out_scale
        pred = int(output_float.flatten()[0] > 0.5)
        if pred == int(y_test[i]):
            correct += 1

    acc = correct / len(X_test)
    print(f"  INT8 TFLite accuracy on test set: {acc:.4f}  "
          f"({correct}/{len(X_test)})\n")
    return acc

def export_c_header(tflite_bytes: bytes,
                    header_path: pathlib.Path,
                    array_name: str = "g_drone_model"):
    header_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(tflite_bytes)

    with open(header_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("// ─── Auto-generated by train_drone_detector.py ───\n")
        f.write(f"// Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"// Model size: {n:,d} bytes ({n/1024:.1f} KB)\n")
        f.write("//\n")
        f.write("// Include this file in your STM32 project and pass\n")
        f.write(f"// {array_name} to tflite::GetModel().\n")
        f.write("//\n\n")
        f.write("#ifndef DRONE_MODEL_H\n")
        f.write("#define DRONE_MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"alignas(16) const uint8_t {array_name}[] = {{\n")

        for i in range(0, n, 12):
            chunk = tflite_bytes[i : i + 12]
            hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
            if i + 12 < n:
                f.write(f"  {hex_vals},\n")
            else:
                f.write(f"  {hex_vals}\n")

        f.write("};\n\n")
        f.write(f"const unsigned int {array_name}_len = {n};\n\n")
        f.write("#endif  // DRONE_MODEL_H\n")

    print(f"  Saved C header → {header_path}")

def export_norm_params(mu: float, std: float, path: pathlib.Path):
    data = {
        "description": "Global log-MFE normalisation for STM32 preprocessing",
        "mean": mu,
        "std": std,
        "formula": "(log10(mel_power + 1e-10) - mean) / std",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved normalisation params → {path}")

    c_path = path.with_suffix(".h")
    with open(c_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("// ─── Auto-generated normalisation constants ───\n")
        f.write("#ifndef NORM_PARAMS_H\n")
        f.write("#define NORM_PARAMS_H\n\n")
        f.write(f"#define MFE_NORM_MEAN  ({mu:.8f}f)\n")
        f.write(f"#define MFE_NORM_STD   ({std:.8f}f)\n\n")
        f.write("#endif  // NORM_PARAMS_H\n")
    print(f"  Saved normalisation header → {c_path}")

def export_dsp_config(path: pathlib.Path):
    cfg = {
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
        "power": POWER,
        "num_frames": NUM_FRAMES,
        "window": "hann",
        "center": False,
        "log_function": "log10(x + 1e-10)",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved DSP config → {path}")

def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("\n" + "═" * 60)
    print("  DRONE ACOUSTIC DETECTOR  –  TinyML Training Pipeline")
    print("  Target: STM32F303RE  |  TFLite Micro + CMSIS-NN")
    print("═" * 60)

    print("\n▸ Step 1/8: Scanning dataset …")
    paths, labels = scan_dataset()

    print("\n▸ Step 2/8: Extracting MFE spectrograms …")
    print(f"    FFT = {FRAME_SIZE},  Hop = {HOP_LENGTH},  "
          f"Mels = {N_MELS},  Frames = {NUM_FRAMES}")
    X, y = build_dataset(paths, labels)
    print(f"    Feature matrix: {X.shape}  (samples, frames, mels)")
    print(f"    Label balance:  drone={int(np.sum(y)):.0f}  "
          f"background={int(len(y) - np.sum(y)):.0f}")

    print("\n▸ Step 3/8: Splitting dataset …")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.15, random_state=RANDOM_SEED, stratify=y_dev
    )

    print("\n▸ Step 4/8: Computing normalisation statistics …")
    norm_mu, norm_std = compute_norm_stats(X_train)
    print(f"    μ = {norm_mu:.6f},  σ = {norm_std:.6f}")

    X_train = apply_norm(X_train, norm_mu, norm_std)
    X_val   = apply_norm(X_val,   norm_mu, norm_std)
    X_test  = apply_norm(X_test,  norm_mu, norm_std)

    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    print(f"    Train: {X_train.shape}  Val: {X_val.shape}  "
          f"Test: {X_test.shape}")

    n_pos = int(np.sum(y_train))
    n_neg = int(len(y_train) - n_pos)
    total = n_pos + n_neg
    class_weight = {
        0: total / (2.0 * n_neg),
        1: total / (2.0 * n_pos),
    }
    print(f"    Class weights: background={class_weight[0]:.3f}  "
          f"drone={class_weight[1]:.3f}")

    print("\n▸ Step 5/8: Building model …")
    input_shape = X_train.shape[1:]       # (NUM_FRAMES, N_MELS, 1)
    model = build_model(input_shape)
    model.summary()

    param_count = model.count_params()
    print(f"\n    Total parameters: {param_count:,d}")
    print(f"    Estimated INT8 weights: ~{param_count / 1024:.1f} KB")

    print("\n▸ Step 6/8: Training …")
    t0 = time.time()
    history = train(model, X_train, y_train, X_val, y_val,
                    class_weight=class_weight)
    elapsed = time.time() - t0
    print(f"\n    Training completed in {elapsed:.1f} s  "
          f"({len(history.history['loss'])} epochs)")

    print("\n▸ Step 7/8: Evaluating …")
    float_metrics = evaluate(model, X_test, y_test, tag="Float32")

    print("\n▸ Step 8/8: INT8 quantisation & export …")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tflite_path = OUTPUT_DIR / "drone_model.tflite"
    tflite_bytes = quantise_int8(model, X_train, tflite_path)

    int8_acc = verify_tflite(tflite_bytes, X_test, y_test,
                              norm_mu, norm_std)

    export_c_header(tflite_bytes, OUTPUT_DIR / "drone_model.h")

    export_norm_params(norm_mu, norm_std,
                       OUTPUT_DIR / "norm_params.json")

    export_dsp_config(OUTPUT_DIR / "dsp_config.json")

    model.save(OUTPUT_DIR / "drone_model_float32.keras")
    print(f"  Saved Keras model → {OUTPUT_DIR / 'drone_model_float32.keras'}")

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Float32 accuracy  : {float_metrics['acc']:.4f}")
    print(f"  INT8 accuracy     : {int8_acc:.4f}")
    print(f"  Model size (Flash): {len(tflite_bytes):,d} bytes "
          f"({len(tflite_bytes)/1024:.1f} KB)")
    print(f"  Output directory  : {OUTPUT_DIR}")
    print()
    print("  Generated files:")
    print(f"    • drone_model.tflite      – TFLite INT8 model")
    print(f"    • drone_model.h           – C uint8_t array for STM32")
    print(f"    • norm_params.json / .h   – Normalisation constants")
    print(f"    • dsp_config.json         – DSP front-end parameters")
    print(f"    • drone_model_float32.keras – Original float model")
    print()
    print("  Next steps:")
    print("    1. Copy drone_model.h and norm_params.h into your")
    print("       STM32CubeIDE project (e.g. Core/Inc/)")
    print("    2. Implement MFE extraction on MCU matching dsp_config.json")
    print("    3. Call tflite::GetModel(g_drone_model) in your firmware")
    print("    4. Allocate tensor_arena[<arena_size>] in RAM")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()
