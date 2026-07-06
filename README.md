# TinyML Acoustic Drone Detector

**Author:** Szymon Warmiak

This project is an AI-based acoustic drone detector running on an **STM32F303RE** microcontroller using **ST X-CUBE-AI**. It uses a lightweight 2D Convolutional Neural Network (CNN) to detect the harmonic acoustic signature of drone propellers.

## Project Structure

* `train_drone_detector.py` - The main training pipeline. Extracts Mel-frequency energy (MFE) spectrograms from raw audio, trains the CNN, performs INT8 quantization, and exports the model.
* `live_test.py` - A live testing tool that uses your PC microphone to test the AI.
* `model_output/` - Contains the trained model (`drone_model.tflite`) and C header files.
* `stm_32_ai_dron_detection/` - **STM32CubeIDE Project**. Contains the complete C firmware running the AI on the MCU.

## STM32 Firmware Features

- **ADC + DMA:** Samples audio from the microphone at 16 kHz continuously using Circular DMA and TIM3 trigger, ensuring 0% CPU usage for acquisition.
- **Optimized DSP:** Precomputed twiddle factors and Hann window for Fast Fourier Transform (FFT). Calculates Mel-spectrogram on the fly in integer bins without floating-point `powf` overhead to prevent HardFaults and fit in the 64ms timing budget.
- **X-CUBE-AI:** Executes the INT8 quantized neural network inference directly on the MCU, consuming only ~22KB of RAM.
- **Hardware Trigger:** Sets GPIO pin PA5 (built-in Nucleo LED) high when a drone is detected.

## Hardware Setup (Microphone + Amplifier)

To get a strong enough signal for the STM32 ADC, the raw electret microphone (e.g., HW-484) needs an amplifier (e.g., LM386).

### Wiring Guide
1. **HW-484 (Mic):** Connect `VCC` to 3.3V, `GND` to GND. Connect `AO` (Analog Out) through a coupling capacitor (e.g., 1µF-10µF) to the `IN` of the LM386.
2. **LM386 (Amplifier):** Connect `VCC` to 5V (Nucleo pin), `GND` to GND.
3. **STM32 ADC (PA0):** Connect LM386 `OUT` through a coupling capacitor (10µF+) to `PA0`. **Crucially**, add a DC bias divider at `PA0` (10kΩ to 3.3V, 10kΩ to GND) to center the audio signal at 1.65V (2048 ADC value). Capacitor polarity: positive (+) leg towards the LM386 (2.5V bias), negative (-) leg towards the STM32 (1.65V bias).

## Dataset

The audio data used to train this model is based on the **Drone Audio Dataset** created by Sara Al-Emadi.
🔗 **[DroneAudioDataset on GitHub](https://github.com/saraalemadi/DroneAudioDataset)**

## Usage (Training)

```bash
pip install -r requirements.txt
python train_drone_detector.py
```
