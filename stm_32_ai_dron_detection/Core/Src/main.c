/**
 * @file    main.c
 * @author  Szymon Warmiak
 * @brief   Drone detection using audio AI on STM32F303RE
 *          ADC1/DMA circular -> FFT -> Mel spectrogram -> X-CUBE-AI inference
 */

#include "main.h"
#include "adc.h"
#include "dma.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* USER CODE BEGIN Includes */
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

void SystemClock_Config(void);

/* USER CODE BEGIN PD */
#define FRAME_SIZE      1024
#define NUM_FRAMES      16
#define N_MELS          20
#define ADC_BUF_SIZE    (FRAME_SIZE * 2)
#define DETECTION_THR   0.5f
#define MFE_NORM_MEAN   (-3.4048095f)
#define MFE_NORM_STD    (3.0600388f)
#define MEL_FMIN        80.0f
#define MEL_FMAX        8000.0f
#define SAMPLE_RATE_F   16000.0f
/* USER CODE END PD */

/* USER CODE BEGIN PV */
static uint16_t  adc_buf[ADC_BUF_SIZE];
static float     fft_re[FRAME_SIZE];
static float     fft_im[FRAME_SIZE];
static float     mel_buf[NUM_FRAMES * N_MELS];

static volatile uint8_t  half_cplt = 0;
static volatile uint8_t  full_cplt = 0;
static          uint8_t  frame_idx = 0;

/* Precomputed tables – filled once in precompute_all() */
static float twiddle_cos[FRAME_SIZE / 2];
static float twiddle_sin[FRAME_SIZE / 2];
static float hann_win[FRAME_SIZE];       /* separate Hann window table */
static int   mel_lo[N_MELS];
static int   mel_ctr[N_MELS];
static int   mel_hi[N_MELS];

/* X-CUBE-AI handles */
static ai_handle          net_handle = AI_HANDLE_NULL;
static ai_u8              net_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static const ai_handle    net_activations_map[] = { (ai_handle)net_activations };
static ai_i8              net_in_buf[AI_NETWORK_IN_1_SIZE];
static ai_i8              net_out_buf[AI_NETWORK_OUT_1_SIZE];
/* USER CODE END PV */

/* USER CODE BEGIN PFP */
static void precompute_all(void);
static void compute_mel_frame(const uint16_t *raw, int offset);
static float run_inference(void);
/* USER CODE END PFP */

/* USER CODE BEGIN 0 */
int __io_putchar(int ch)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

static void precompute_all(void)
{
    /* FFT twiddle factors */
    for (int i = 0; i < FRAME_SIZE / 2; i++) {
        twiddle_cos[i] = cosf(-2.0f * 3.14159265f * i / FRAME_SIZE);
        twiddle_sin[i] = sinf(-2.0f * 3.14159265f * i / FRAME_SIZE);
    }

    /* Hann window (periodic, matches librosa default) */
    for (int i = 0; i < FRAME_SIZE; i++) {
        hann_win[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * i / FRAME_SIZE));
    }

    /* Mel filter bank – triangle boundaries as integer bin indices */
    float fmin_mel = 2595.0f * log10f(1.0f + MEL_FMIN / 700.0f);
    float fmax_mel = 2595.0f * log10f(1.0f + MEL_FMAX / 700.0f);

    float hz[N_MELS + 2];
    for (int i = 0; i <= N_MELS + 1; i++) {
        float mel = fmin_mel + (float)i * (fmax_mel - fmin_mel) / (N_MELS + 1);
        hz[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    for (int m = 0; m < N_MELS; m++) {
        mel_lo[m]  = (int)(hz[m]   * FRAME_SIZE / SAMPLE_RATE_F);
        mel_ctr[m] = (int)(hz[m+1] * FRAME_SIZE / SAMPLE_RATE_F);
        mel_hi[m]  = (int)(hz[m+2] * FRAME_SIZE / SAMPLE_RATE_F);
        if (mel_hi[m] >= FRAME_SIZE / 2) mel_hi[m] = FRAME_SIZE / 2 - 1;
    }

    printf("Tables ready. Mel bins 0-1: [%d-%d-%d], [%d-%d-%d]\r\n",
           mel_lo[0], mel_ctr[0], mel_hi[0],
           mel_lo[1], mel_ctr[1], mel_hi[1]);
}

static void compute_mel_frame(const uint16_t *raw, int offset)
{
    /* Remove DC bias and apply Hann window */
    float dc = 0.0f;
    for (int i = 0; i < FRAME_SIZE; i++)
        dc += (float)raw[offset + i];
    dc /= FRAME_SIZE;

    for (int i = 0; i < FRAME_SIZE; i++) {
        fft_re[i] = (((float)raw[offset + i] - dc) / 2048.0f) * hann_win[i];
        fft_im[i] = 0.0f;
    }

    /* In-place radix-2 DIT FFT using precomputed twiddle table */
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < FRAME_SIZE; i++) {
        int bit = FRAME_SIZE >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = fft_re[i]; fft_re[i] = fft_re[j]; fft_re[j] = tr;
            float ti = fft_im[i]; fft_im[i] = fft_im[j]; fft_im[j] = ti;
        }
    }
    /* Butterfly stages */
    for (int s = 1; s < FRAME_SIZE; s <<= 1) {
        int half_s = s;
        int step   = FRAME_SIZE / (2 * s);
        for (int k = 0; k < FRAME_SIZE; k += 2 * half_s) {
            for (int j = 0; j < half_s; j++) {
                float wr = twiddle_cos[j * step];
                float wi = twiddle_sin[j * step];
                float tr = wr * fft_re[k+j+half_s] - wi * fft_im[k+j+half_s];
                float ti = wr * fft_im[k+j+half_s] + wi * fft_re[k+j+half_s];
                fft_re[k+j+half_s] = fft_re[k+j] - tr;
                fft_im[k+j+half_s] = fft_im[k+j] - ti;
                fft_re[k+j] += tr;
                fft_im[k+j] += ti;
            }
        }
    }

    /* Mel filter bank – no powf here, just integer bin loops */
    int row = frame_idx * N_MELS;
    for (int m = 0; m < N_MELS; m++) {
        float energy = 0.0f;
        int lo = mel_lo[m], ctr = mel_ctr[m], hi = mel_hi[m];
        if (ctr <= lo) ctr = lo + 1;
        if (hi  <= ctr) hi  = ctr + 1;
        for (int b = lo; b < ctr; b++) {
            float w   = (float)(b - lo) / (float)(ctr - lo);
            float mag = fft_re[b]*fft_re[b] + fft_im[b]*fft_im[b];
            energy   += w * mag;
        }
        for (int b = ctr; b <= hi; b++) {
            float w   = (float)(hi - b) / (float)(hi - ctr);
            float mag = fft_re[b]*fft_re[b] + fft_im[b]*fft_im[b];
            energy   += w * mag;
        }
        mel_buf[row + m] = (log10f(energy + 1e-10f) - MFE_NORM_MEAN) / MFE_NORM_STD;
    }

    frame_idx++;
}

static float run_inference(void)
{
    ai_buffer input_buf  = AI_BUFFER_INIT(AI_FLAG_NONE, AI_BUFFER_FORMAT_S8,
                                          AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1, 20, 16),
                                          AI_NETWORK_IN_1_SIZE, NULL, net_in_buf);
    ai_buffer output_buf = AI_BUFFER_INIT(AI_FLAG_NONE, AI_BUFFER_FORMAT_S8,
                                          AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1, 1, 1),
                                          AI_NETWORK_OUT_1_SIZE, NULL, net_out_buf);

    ai_buffer *inputs  = ai_network_inputs_get(net_handle, NULL);
    ai_buffer *outputs = ai_network_outputs_get(net_handle, NULL);

    float   in_scale  = 1.0f;
    int32_t in_zp     = 0;
    float   out_scale = 1.0f;
    int32_t out_zp    = 0;

    if (inputs && inputs->meta_info && inputs->meta_info->intq_info) {
        in_scale = inputs->meta_info->intq_info->info[0].scale[0];
        in_zp    = *((const int8_t*)inputs->meta_info->intq_info->info[0].zeropoint);
    }
    if (outputs && outputs->meta_info && outputs->meta_info->intq_info) {
        out_scale = outputs->meta_info->intq_info->info[0].scale[0];
        out_zp    = *((const int8_t*)outputs->meta_info->intq_info->info[0].zeropoint);
    }

    /* Debug: print mel stats (use int/frac trick for nano.specs which lacks %f) */
    float mel_min = mel_buf[0], mel_max = mel_buf[0], mel_sum = 0.0f;
    for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
        if (mel_buf[i] < mel_min) mel_min = mel_buf[i];
        if (mel_buf[i] > mel_max) mel_max = mel_buf[i];
        mel_sum += mel_buf[i];
    }
    float mel_avg = mel_sum / AI_NETWORK_IN_1_SIZE;
    /* Print float as X.YY (sign + integer + 2 decimals) */
    #define PFLT(v) (int)(v), (int)(((v) - (int)(v)) * 100 + 0.5f)
    printf("[MEL] min=%d.%02d max=%d.%02d avg=%d.%02d sc=%d.%04d\r\n",
           (int)mel_min,    (int)((mel_min    - (int)mel_min)    * 100 + 0.5f),
           (int)mel_max,    (int)((mel_max    - (int)mel_max)    * 100 + 0.5f),
           (int)mel_avg,    (int)((mel_avg    - (int)mel_avg)    * 100 + 0.5f),
           (int)in_scale,   (int)((in_scale   - (int)in_scale)   * 10000 + 0.5f));

    for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
        float q = mel_buf[i] / in_scale + (float)in_zp;
        if      (q >  127.0f) q =  127.0f;
        else if (q < -128.0f) q = -128.0f;
        net_in_buf[i] = (ai_i8)(int8_t)q;
    }


    ai_i32 n_batch = ai_network_run(net_handle, &input_buf, &output_buf);
    if (n_batch != 1) return 0.0f;

    /* Model output = P(background). Invert to get P(drone). */
    float conf = ((float)net_out_buf[0] - (float)out_zp) * out_scale;
    conf = 1.0f - conf;   /* invert: high = drone */
    if (conf < 0.0f) conf = 0.0f;
    if (conf > 1.0f) conf = 1.0f;
    return conf;
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    (void)hadc;
    half_cplt = 1;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    (void)hadc;
    full_cplt = 1;
}
/* USER CODE END 0 */

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_ADC1_Init();
    MX_TIM3_Init();
    MX_USART2_UART_Init();

    /* USER CODE BEGIN 2 */
    printf("Drone Detector starting...\r\n");
    precompute_all();
    ai_network_create_and_init(&net_handle, net_activations_map, NULL);
    printf("AI ready.\r\n");

    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buf, ADC_BUF_SIZE);
    HAL_TIM_Base_Start(&htim3);
    /* USER CODE END 2 */

    while (1)
    {
        /* USER CODE BEGIN 3 */
        if (half_cplt) {
            half_cplt = 0;
            if (frame_idx < NUM_FRAMES)
                compute_mel_frame(adc_buf, 0);
        }
        if (full_cplt) {
            full_cplt = 0;
            if (frame_idx < NUM_FRAMES)
                compute_mel_frame(adc_buf, FRAME_SIZE);

            if (frame_idx >= NUM_FRAMES) {
                frame_idx = 0;
                float conf = run_inference();
                if (conf >= DETECTION_THR) {
                    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
                    printf("DRON WYKRYTY! %d%%\r\n", (int)(conf * 100));
                } else {
                    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
                    printf("Brak drona.  %d%%\r\n", (int)(conf * 100));
                }
            }
        }
        /* USER CODE END 3 */
    }
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState            = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLMUL          = RCC_PLL_MUL9;
    RCC_OscInitStruct.PLL.PREDIV          = RCC_PREDIV_DIV1;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) Error_Handler();

    RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                     |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) Error_Handler();

    PeriphClkInit.PeriphClockSelection  = RCC_PERIPHCLK_USART2|RCC_PERIPHCLK_ADC12
                                        |RCC_PERIPHCLK_TIM34;
    PeriphClkInit.Usart2ClockSelection  = RCC_USART2CLKSOURCE_PCLK1;
    PeriphClkInit.Adc12ClockSelection   = RCC_ADC12PLLCLK_DIV1;
    PeriphClkInit.Tim34ClockSelection   = RCC_TIM34CLK_HCLK;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK) Error_Handler();
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif
