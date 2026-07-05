# TinyML Acoustic Drone Detector

This project is an AI-based acoustic drone detector designed to run on microcontrollers (e.g., STM32) using TensorFlow Lite Micro. It uses a lightweight 2D Convolutional Neural Network (CNN) to detect the harmonic acoustic signature of drone propellers.

## Project Structure

* `train_drone_detector.py` - The main training pipeline. It extracts Mel-frequency energy (MFE) spectrograms from raw audio, trains the CNN, performs INT8 post-training quantization, and exports the model as a `.tflite` file and a C header for microcontrollers.
* `live_test.py` - A live testing tool that uses your computer's microphone to visualize audio amplitude, FFT spectrum, Mel-spectrogram, and real-time AI drone detection confidence.
* `model_output/` - Contains the trained model (`drone_model.tflite`), the exported C array (`drone_model.h`), and DSP configuration/normalization parameters.

## Dataset

The audio data used to train this model is based on the **Drone Audio Dataset** created by Sara Al-Emadi.
You can find the original dataset and more information here:
🔗 **[DroneAudioDataset on GitHub](https://github.com/saraalemadi/DroneAudioDataset)**

## Usage

### 1. Training the Model
To re-train the model from scratch, install the required dependencies and run the training script:
```bash
pip install -r requirements.txt
python train_drone_detector.py
```

### 2. Live Microphone Test
To test the trained model in real-time using your computer's microphone:
```bash
pip install sounddevice matplotlib
python live_test.py
```
*(If you have multiple microphones, you can pass the device ID as an argument: `python live_test.py 2`)*
