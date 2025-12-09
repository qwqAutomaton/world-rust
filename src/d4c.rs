use crate::common::utils::{EPS, dc_fix, interp1, linear_smoothing};
use crate::common::window::nuttall_inplace;
use crate::dio::F0Contour;
use num_complex::Complex64;
use rand::Rng;
use realfft::{RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::sync::Arc;

const FREQUENCY_INTERVAL: f64 = 3000.0;
const UPPER_LIMIT: f64 = 15000.0;
const FLOOR_F0_D4C: f64 = 47.0;
const LOWEST_F0: f64 = 40.0;
const WINDOW_RATIO_GENERAL: f64 = 4.0;
const WINDOW_RATIO_LOVE: f64 = 3.0;
const HANNING_SHIFT: f64 = 0.5;

pub fn D4C(
    x: &[f64],
    fs: u32,
    f0_contour: &F0Contour,
    fft_size: usize,
    threshold: f64,
) -> Vec<Vec<f64>> {
    let f0_length = f0_contour.f0.len();
    if f0_length == 0 {
        return Vec::new();
    }
    assert_eq!(
        f0_length,
        f0_contour.tpos.len(),
        "F0 contour length mismatch"
    );

    let spectrum_length = fft_size / 2 + 1;
    let mut aperiodicity = vec![vec![1.0 - EPS; spectrum_length]; f0_length];

    let fft_size_d4c = fft_size_from_multiplier(fs, 4.0, FLOOR_F0_D4C);
    let love_fft_size = fft_size_from_multiplier(fs, 3.0, LOWEST_F0);

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c_d4c = planner.plan_fft_forward(fft_size_d4c);
    let r2c_love = planner.plan_fft_forward(love_fft_size);

    let mut fft_buffers = ForwardFftBuffers::new(fft_size_d4c, &r2c_d4c);
    let mut love_buffers = ForwardFftBuffers::new(love_fft_size, &r2c_love);
    let mut rng = rand::rng();

    let mut aperiodicity0 = vec![0.0; f0_length];
    d4c_love_train(
        x,
        fs,
        &f0_contour.f0,
        &f0_contour.tpos,
        &mut aperiodicity0,
        &r2c_love,
        &mut love_buffers,
        &mut rng,
    );

    let usable_upper = (fs as f64 / 2.0 - FREQUENCY_INTERVAL).min(UPPER_LIMIT);
    let number_of_aperiodicities = (usable_upper / FREQUENCY_INTERVAL).floor() as usize;

    let window_length = ((FREQUENCY_INTERVAL * fft_size_d4c as f64 / fs as f64) as usize) * 2 + 1;
    let mut window = vec![0.0; window_length];
    nuttall_inplace(window_length, &mut window);

    let mut coarse_frequency_axis = vec![0.0; number_of_aperiodicities + 2];
    for i in 0..=number_of_aperiodicities {
        coarse_frequency_axis[i] = i as f64 * FREQUENCY_INTERVAL;
    }
    coarse_frequency_axis[number_of_aperiodicities + 1] = fs as f64 / 2.0;

    let frequency_axis: Vec<f64> = (0..=fft_size / 2)
        .map(|i| i as f64 * fs as f64 / fft_size as f64)
        .collect();

    let mut coarse_aperiodicity = vec![0.0; number_of_aperiodicities + 2];
    coarse_aperiodicity[0] = -60.0;
    coarse_aperiodicity[number_of_aperiodicities + 1] = -EPS;

    for ((&raw_f0, &tpos), (aperiodicity_row, ap0)) in f0_contour
        .f0
        .iter()
        .zip(&f0_contour.tpos)
        .zip(aperiodicity.iter_mut().zip(&aperiodicity0))
    {
        if raw_f0 == 0.0 || *ap0 <= threshold {
            continue;
        }
        let current_f0 = raw_f0.max(FLOOR_F0_D4C);

        d4c_general_body(
            x,
            fs,
            current_f0,
            fft_size_d4c,
            tpos,
            &window,
            &r2c_d4c,
            &mut fft_buffers,
            &mut coarse_aperiodicity[1..=number_of_aperiodicities],
            &mut rng,
        );

        get_aperiodicity(
            &coarse_frequency_axis,
            &coarse_aperiodicity,
            &frequency_axis,
            aperiodicity_row,
        );
    }

    aperiodicity
}

enum WindowType {
    Hanning,
    Blackman,
}

struct ForwardFftBuffers {
    waveform: Vec<f64>,
    spectrum: Vec<Complex64>,
}

impl ForwardFftBuffers {
    fn new(fft_size: usize, r2c: &Arc<dyn RealToComplex<f64>>) -> Self {
        Self {
            waveform: vec![0.0; fft_size],
            spectrum: r2c.make_output_vec(),
        }
    }

    fn reset(&mut self) {
        self.waveform.fill(0.0);
    }

    fn fft_size(&self) -> usize {
        self.waveform.len()
    }

    fn spectrum_len(&self) -> usize {
        self.spectrum.len()
    }
}

fn fft_size_from_multiplier(fs: u32, multiplier: f64, floor_f0: f64) -> usize {
    let value = multiplier * fs as f64 / floor_f0 + 1.0;
    let exponent = value.log2().floor().max(0.0) as u32;
    2_usize.pow(1 + exponent)
}

fn sample_normal(rng: &mut impl Rng) -> f64 {
    let u1 = rng.random::<f64>().clamp(EPS, 1.0 - EPS);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn get_windowed_waveform(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    current_position: f64,
    window_type: WindowType,
    window_length_ratio: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut half_window_length =
        (window_length_ratio * fs as f64 / current_f0 / 2.0).round() as isize;
    if half_window_length < 1 {
        half_window_length = 1;
    }
    let window_length = (2 * half_window_length + 1) as usize;
    let origin = (current_position * fs as f64 + 0.001).round() as isize;

    let mut waveform = vec![0.0; window_length];
    let mut window = vec![0.0; window_length];

    for i in 0..window_length {
        let base = i as isize - half_window_length;
        let safe_idx = (origin + base).clamp(0, (x.len() as isize - 1).max(0)) as usize;
        let position = (2.0 * base as f64 / window_length_ratio) / fs as f64;
        let window_value = match window_type {
            WindowType::Hanning => 0.5 * (PI * position * current_f0).cos() + HANNING_SHIFT,
            WindowType::Blackman => {
                let argument = PI * position * current_f0;
                0.42 + 0.5 * argument.cos() + 0.08 * (2.0 * argument).cos()
            }
        };
        window[i] = window_value;
        waveform[i] = x[safe_idx] * window_value + sample_normal(rng) * EPS;
    }

    let sum_waveform: f64 = waveform.iter().sum();
    let sum_window: f64 = window.iter().sum();
    let weighting = sum_waveform / (sum_window + EPS);
    for (sample, win) in waveform.iter_mut().zip(window.iter()) {
        *sample -= win * weighting;
    }

    waveform
}

fn copy_into_buffers(buffers: &mut ForwardFftBuffers, data: &[f64]) {
    buffers.reset();
    let len = data.len().min(buffers.waveform.len());
    buffers.waveform[..len].copy_from_slice(&data[..len]);
}

fn forward_fft(buffers: &mut ForwardFftBuffers, r2c: &Arc<dyn RealToComplex<f64>>) {
    r2c.process(&mut buffers.waveform, &mut buffers.spectrum)
        .expect("real FFT forward transform failed");
}

fn get_centroid(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    current_position: f64,
    buffers: &mut ForwardFftBuffers,
    r2c: &Arc<dyn RealToComplex<f64>>,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut waveform = get_windowed_waveform(
        x,
        fs,
        current_f0,
        current_position,
        WindowType::Blackman,
        WINDOW_RATIO_GENERAL,
        rng,
    );

    let limit = (((2.0 * fs as f64 / current_f0).round() as usize) * 2)
        .min(waveform.len().saturating_sub(1));
    let mut power = 0.0;
    for i in 0..=limit {
        power += waveform[i] * waveform[i];
    }
    let norm = power.max(EPS).sqrt();
    for i in 0..=limit {
        waveform[i] /= norm;
    }

    copy_into_buffers(buffers, &waveform);
    forward_fft(buffers, r2c);

    let tmp_real: Vec<f64> = buffers.spectrum.iter().map(|c| c.re).collect();
    let tmp_imag: Vec<f64> = buffers.spectrum.iter().map(|c| c.im).collect();

    for (i, value) in waveform.iter_mut().enumerate() {
        *value *= (i + 1) as f64;
    }

    copy_into_buffers(buffers, &waveform);
    forward_fft(buffers, r2c);

    buffers
        .spectrum
        .iter()
        .zip(tmp_real.iter().zip(tmp_imag.iter()))
        .map(|(spectrum, (real0, imag0))| spectrum.re * real0 + imag0 * spectrum.im)
        .collect()
}

fn get_static_centroid(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    buffers: &mut ForwardFftBuffers,
    r2c: &Arc<dyn RealToComplex<f64>>,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let centroid1 = get_centroid(
        x,
        fs,
        current_f0,
        current_position - 0.25 / current_f0,
        buffers,
        r2c,
        rng,
    );
    let centroid2 = get_centroid(
        x,
        fs,
        current_f0,
        current_position + 0.25 / current_f0,
        buffers,
        r2c,
        rng,
    );

    let mut combined: Vec<f64> = centroid1
        .into_iter()
        .zip(centroid2)
        .map(|(c1, c2)| c1 + c2)
        .collect();

    combined = dc_fix(&combined, current_f0, fs, fft_size);
    combined
}

fn get_smoothed_power_spectrum(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    buffers: &mut ForwardFftBuffers,
    r2c: &Arc<dyn RealToComplex<f64>>,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let waveform = get_windowed_waveform(
        x,
        fs,
        current_f0,
        current_position,
        WindowType::Hanning,
        WINDOW_RATIO_GENERAL,
        rng,
    );

    copy_into_buffers(buffers, &waveform);
    forward_fft(buffers, r2c);

    let mut power: Vec<f64> = buffers
        .spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();

    power = dc_fix(&power, current_f0, fs, fft_size);
    linear_smoothing(&power, current_f0, fs, fft_size)
}

fn get_static_group_delay(
    static_centroid: &[f64],
    smoothed_power_spectrum: &[f64],
    fs: u32,
    f0: f64,
    fft_size: usize,
) -> Vec<f64> {
    let mut static_group_delay: Vec<f64> = static_centroid
        .iter()
        .zip(smoothed_power_spectrum.iter())
        .map(|(c, s)| c / (s + EPS))
        .collect();

    static_group_delay = linear_smoothing(&static_group_delay, f0 / 2.0, fs, fft_size);
    let smoothed_group_delay = linear_smoothing(&static_group_delay, f0, fs, fft_size);

    static_group_delay
        .iter_mut()
        .zip(smoothed_group_delay)
        .for_each(|(value, smooth)| *value -= smooth);

    static_group_delay
}

fn get_coarse_aperiodicity(
    static_group_delay: &[f64],
    fs: u32,
    fft_size: usize,
    window: &[f64],
    buffers: &mut ForwardFftBuffers,
    r2c: &Arc<dyn RealToComplex<f64>>,
    coarse_aperiodicity: &mut [f64],
) {
    let half_window_length = window.len() / 2;
    let boundary = ((fft_size as f64 * 8.0 / window.len() as f64).round() as usize)
        .min(buffers.spectrum_len() - 1);
    let mut power_spectrum = vec![0.0; buffers.spectrum_len()];

    for (band_index, value) in coarse_aperiodicity.iter_mut().enumerate() {
        let center =
            ((FREQUENCY_INTERVAL * (band_index + 1) as f64 * fft_size as f64) / fs as f64) as isize;
        let start = center - half_window_length as isize;
        if start < 0 || (start as usize + window.len()) > static_group_delay.len() {
            *value = -60.0;
            continue;
        }

        buffers.reset();
        for j in 0..window.len() {
            let idx = start as usize + j;
            buffers.waveform[j] = static_group_delay[idx] * window[j];
        }

        forward_fft(buffers, r2c);
        for (dst, src) in power_spectrum.iter_mut().zip(buffers.spectrum.iter()) {
            *dst = src.re * src.re + src.im * src.im;
        }

        power_spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 1..power_spectrum.len() {
            let prev = power_spectrum[i - 1];
            power_spectrum[i] += prev;
        }

        let total_index = power_spectrum.len() - 1;
        let lower_index = total_index.saturating_sub(boundary + 1);
        let numerator = power_spectrum[lower_index].max(EPS);
        let denominator = power_spectrum[total_index].max(EPS);
        *value = 10.0 * (numerator / denominator).log10();
    }
}

fn d4c_general_body(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    fft_size: usize,
    current_position: f64,
    window: &[f64],
    r2c: &Arc<dyn RealToComplex<f64>>,
    buffers: &mut ForwardFftBuffers,
    coarse_aperiodicity: &mut [f64],
    rng: &mut impl Rng,
) {
    let static_centroid = get_static_centroid(
        x,
        fs,
        current_f0,
        fft_size,
        current_position,
        buffers,
        r2c,
        rng,
    );
    let smoothed_power_spectrum = get_smoothed_power_spectrum(
        x,
        fs,
        current_f0,
        fft_size,
        current_position,
        buffers,
        r2c,
        rng,
    );
    let static_group_delay = get_static_group_delay(
        &static_centroid,
        &smoothed_power_spectrum,
        fs,
        current_f0,
        fft_size,
    );

    get_coarse_aperiodicity(
        &static_group_delay,
        fs,
        fft_size,
        window,
        buffers,
        r2c,
        coarse_aperiodicity,
    );

    for value in coarse_aperiodicity.iter_mut() {
        *value = (*value + (current_f0 - 100.0) / 50.0).min(0.0);
    }
}

fn get_aperiodicity(
    coarse_frequency_axis: &[f64],
    coarse_aperiodicity: &[f64],
    frequency_axis: &[f64],
    output: &mut [f64],
) {
    let interpolated = interp1(coarse_frequency_axis, coarse_aperiodicity, frequency_axis);
    for (dst, value) in output.iter_mut().zip(interpolated) {
        *dst = 10f64.powf(value / 20.0);
    }
}

fn d4c_love_train(
    x: &[f64],
    fs: u32,
    f0: &[f64],
    temporal_positions: &[f64],
    aperiodicity0: &mut [f64],
    r2c: &Arc<dyn RealToComplex<f64>>,
    buffers: &mut ForwardFftBuffers,
    rng: &mut impl Rng,
) {
    let fft_size = buffers.fft_size();
    let boundary0 =
        ((100.0 * fft_size as f64 / fs as f64).ceil() as usize).min(buffers.spectrum_len() - 1);
    let boundary1 =
        ((4000.0 * fft_size as f64 / fs as f64).ceil() as usize).min(buffers.spectrum_len() - 1);
    let boundary2 =
        ((7900.0 * fft_size as f64 / fs as f64).ceil() as usize).min(buffers.spectrum_len() - 1);

    for ((aperiodicity, &current_f0), &position) in aperiodicity0
        .iter_mut()
        .zip(f0.iter())
        .zip(temporal_positions.iter())
    {
        if current_f0 == 0.0 {
            *aperiodicity = 0.0;
            continue;
        }

        *aperiodicity = d4c_love_train_sub(
            x,
            fs,
            current_f0.max(LOWEST_F0),
            position,
            boundary0,
            boundary1,
            boundary2,
            r2c,
            buffers,
            rng,
        );
    }
}

fn d4c_love_train_sub(
    x: &[f64],
    fs: u32,
    current_f0: f64,
    current_position: f64,
    boundary0: usize,
    boundary1: usize,
    boundary2: usize,
    r2c: &Arc<dyn RealToComplex<f64>>,
    buffers: &mut ForwardFftBuffers,
    rng: &mut impl Rng,
) -> f64 {
    let window = get_windowed_waveform(
        x,
        fs,
        current_f0,
        current_position,
        WindowType::Blackman,
        WINDOW_RATIO_LOVE,
        rng,
    );

    copy_into_buffers(buffers, &window);
    forward_fft(buffers, r2c);

    let mut power_spectrum: Vec<f64> = buffers
        .spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();

    for value in power_spectrum.iter_mut().take(boundary0 + 1) {
        *value = 0.0;
    }

    for i in (boundary0 + 1)..power_spectrum.len() {
        let magnitude = buffers.spectrum[i];
        power_spectrum[i] = magnitude.re * magnitude.re + magnitude.im * magnitude.im;
    }

    for i in (boundary0 + 1)..=boundary2.min(power_spectrum.len() - 1) {
        power_spectrum[i] += power_spectrum[i - 1];
    }

    let numerator = power_spectrum[boundary1.min(power_spectrum.len() - 1)].max(EPS);
    let denominator = power_spectrum[boundary2.min(power_spectrum.len() - 1)].max(EPS);
    numerator / denominator
}
