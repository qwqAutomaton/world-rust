use crate::common::utils::{self, EPS, linear_smoothing};
use crate::dio;
use rand::Rng;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const DEFAULT_F0: f64 = 500.0;

fn random_float(rng: &mut impl Rng) -> f64 {
    rng.random()
}

pub fn cheaptrick(
    samples: &[f64],
    fs: u32,
    f0_contour: &dio::F0Contour,
    option: &CheapTrickOption,
) -> Vec<Vec<f64>> {
    let fft_size = match option.fftsize {
        0 => get_fftsize(fs, option.f0_floor),
        _ => option.fftsize,
    };
    let f0_floor = get_f0floor(fs, fft_size);
    // reuse fft planner
    let mut planner = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    // result
    let mut spectrogram = vec![vec![0.0; fft_size / 2 + 1]; f0_contour.f0.len()];
    // random generator
    let mut rng = rand::rng();
    // do cheaptrick frame-wise
    for (i, &raw_f0) in f0_contour.f0.iter().enumerate() {
        // limit f0 to default
        let current_f0 = if raw_f0 <= f0_floor {
            DEFAULT_F0
        } else {
            raw_f0
        };
        // perform frame-wise cheaptrick
        cheaptrick_frame(
            samples,
            fs,
            current_f0,
            f0_contour.tpos[i],
            option.q1,
            fft_size,
            &r2c,
            &c2r,
            &mut rng,
            &mut spectrogram[i],
        );
    }

    spectrogram
}

pub struct CheapTrickOption {
    pub q1: f64,
    pub f0_floor: f64,
    pub fftsize: usize,
}

impl Default for CheapTrickOption {
    fn default() -> Self {
        Self {
            q1: -0.15,
            f0_floor: 71.0,
            fftsize: 0, // 必须手动设置
        }
    }
}

pub fn get_fftsize(fs: u32, f0_floor: f64) -> usize {
    2_usize.pow(1 + (3.0 * fs as f64 / f0_floor + 1.0).log2() as u32)
}

pub fn get_f0floor(fs: u32, fft_size: usize) -> f64 {
    3.0 * fs as f64 / (fft_size as f64 - 3.0)
}

// frame-wise cheaptrick
fn cheaptrick_frame(
    x: &[f64],             // raw samples
    fs: u32,               // sample rate
    current_f0: f64,       // f0 calculated by DIO
    current_position: f64, // temporal position
    q1: f64,               // default = -0.15
    fft_size: usize,
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
    rng: &mut impl Rng,
    spectral_envelope: &mut [f64], // inplace result
) {
    // get windowed samples (length ~ 3*fs/f0)
    let mut waveform = get_windowed_waveform(x, fs, current_f0, current_position, fft_size, rng);

    // get power spectrum (+ DC fix)
    let power_spectrum = get_power_spectrum(fs, current_f0, fft_size, &mut waveform, r2c);

    // Smoothing of the power (linear axis)
    let smoothed_power_spectrum =
        linear_smoothing(&power_spectrum, current_f0 * 2.0 / 3.0, fs, fft_size);

    // prevent log(0)
    let mut power_spectrum_for_log = smoothed_power_spectrum
        .iter()
        .map(|&val| val + random_float(rng).abs() * EPS)
        .collect::<Vec<_>>();

    // Smoothing (log axis) and spectral recovery on the cepstrum domain.
    smoothing_with_recovery(
        &mut power_spectrum_for_log,
        current_f0,
        fs,
        fft_size,
        q1,
        r2c,
        c2r,
        spectral_envelope,
    );
}

// C++: GetWindowedWaveform
fn get_windowed_waveform(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    current_position: f64,
    fft_size: usize,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let half_window_length = (1.5 * fs as f64 / current_f0).round() as usize;
    let window_length = 2 * half_window_length + 1;

    let (_base_index, safe_index, window) = set_parameters_for_get_windowed_waveform(
        half_window_length,
        x.len(),
        current_position,
        fs,
        current_f0,
    );

    let mut waveform = vec![0.0; fft_size];
    for i in 0..window_length {
        waveform[i] = x[safe_index[i]] * window[i] + random_float(rng) * EPS;
    }

    let tmp_weight1: f64 = waveform.iter().take(window_length).sum();
    let tmp_weight2: f64 = window.iter().sum();
    let weighting_coefficient = tmp_weight1 / tmp_weight2;

    for i in 0..window_length {
        waveform[i] -= window[i] * weighting_coefficient;
    }

    waveform
}

// C++: SetParametersForGetWindowedWaveform
fn set_parameters_for_get_windowed_waveform(
    half_window_length: usize,
    x_length: usize,
    current_position: f64,
    fs: u32,
    current_f0: f64,
) -> (Vec<i32>, Vec<usize>, Vec<f64>) {
    let window_length = 2 * half_window_length + 1;
    let base_index: Vec<i32> =
        (-(half_window_length as i32)..=(half_window_length as i32)).collect();
    let origin = (current_position * fs as f64 + 0.001).round() as isize;

    let safe_index: Vec<usize> = (0..window_length)
        .map(|i| (origin + base_index[i] as isize).clamp(0, (x_length - 1) as isize) as usize)
        .collect();

    let mut window = vec![0.0; window_length];
    let mut average = 0.0;
    for i in 0..window_length {
        let position = base_index[i] as f64 / (1.5 * fs as f64);
        window[i] = 0.5 * (std::f64::consts::PI * position * current_f0).cos() + 0.5;
        average += window[i] * window[i];
    }
    average = average.sqrt();
    for w in &mut window {
        *w /= average;
    }

    (base_index, safe_index, window)
}

// C++: GetPowerSpectrum
fn get_power_spectrum(
    fs: u32,
    f0: f64,
    fft_size: usize,
    waveform: &mut [f64],
    r2c: &Arc<dyn RealToComplex<f64>>,
) -> Vec<f64> {
    let mut spectrum = r2c.make_output_vec();
    r2c.process(waveform, &mut spectrum).unwrap();

    let mut power_spectrum = vec![0.0; fft_size / 2 + 1];
    for i in 0..=fft_size / 2 {
        power_spectrum[i] = spectrum[i].re.powi(2) + spectrum[i].im.powi(2);
    }

    utils::dc_fix(&power_spectrum, f0, fs, fft_size)
}

// C++: SmoothingWithRecovery
fn smoothing_with_recovery(
    power_spectrum: &mut [f64],
    f0: f64,
    fs: u32,
    fft_size: usize,
    q1: f64,
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
    spectral_envelope: &mut [f64],
) {
    let mut smoothing_lifter = vec![0.0; fft_size / 2 + 1];
    let mut compensation_lifter = vec![0.0; fft_size / 2 + 1];

    smoothing_lifter[0] = 1.0;
    compensation_lifter[0] = 1.0;

    for i in 1..=fft_size / 2 {
        let quefrency = i as f64 / fs as f64;
        let angular_freq = std::f64::consts::PI * f0 * quefrency;
        smoothing_lifter[i] = (angular_freq).sin() / angular_freq;
        compensation_lifter[i] =
            (1.0 - 2.0 * q1) + 2.0 * q1 * (2.0 * std::f64::consts::PI * quefrency * f0).cos();
    }

    let log_spectrum: Vec<f64> = power_spectrum.iter().map(|&x| x.ln()).collect();

    // Mirroring for RFFT
    let mut log_spectrum_for_fft = r2c.make_input_vec();
    log_spectrum_for_fft[..log_spectrum.len()].copy_from_slice(&log_spectrum);
    for i in 1..fft_size / 2 {
        log_spectrum_for_fft[fft_size - i] = log_spectrum_for_fft[i];
    }

    let mut cepstrum = r2c.make_output_vec();
    r2c.process(&mut log_spectrum_for_fft, &mut cepstrum)
        .unwrap();

    // Apply lifters
    for i in 0..=fft_size / 2 {
        cepstrum[i] *= smoothing_lifter[i] * compensation_lifter[i] / fft_size as f64;
    }

    // According to FFTW's documentation for r2c, the 0-th and N/2-th components
    // are not multiplied by 2.
    // for i in 1..fft_size / 2 {
    //     cepstrum[i] *= 2.0;
    // }

    let mut smoothed_log_spectrum = c2r.make_output_vec();
    c2r.process(&mut cepstrum, &mut smoothed_log_spectrum)
        .unwrap();

    for i in 0..=fft_size / 2 {
        spectral_envelope[i] = smoothed_log_spectrum[i].exp();
    }
}
