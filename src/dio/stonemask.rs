use std::{f64::consts::PI, sync::Arc};

use crate::common::window::{blackman_inplace, differential_inplace};
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

const STONEMASK_F0_FLOOR: f64 = 40.0;
const STONEMASK_THRESHOLD: f64 = 0.2;

/// StoneMask 优化整条基频轮廓（复用 FFT 计划与缓冲，减少分配）
pub fn refine_f0_contour(x: &[f64], f0: &[f64], tpos: &[f64], fs: u32) -> Vec<f64> {
    let mut planner = RealFftPlanner::<f64>::new();
    let mut scratch = StoneMaskScratch::new();
    f0.iter()
        .zip(tpos.iter())
        .map(|(&f, &t)| refine_f0_with(&mut planner, &mut scratch, x, f, t, fs))
        .collect()
}

/// 优化一帧 f0
pub fn refine_f0(x: &[f64], f0: f64, tpos: f64, fs: u32) -> f64 {
    // 非批量路径：单次使用也能工作，但不复用 planner 与缓冲。
    let mut planner = RealFftPlanner::<f64>::new();
    let mut scratch = StoneMaskScratch::new();
    refine_f0_with(&mut planner, &mut scratch, x, f0, tpos, fs)
}

fn refine_f0_with(
    planner: &mut RealFftPlanner<f64>,
    scratch: &mut StoneMaskScratch,
    x: &[f64],
    f0: f64,
    tpos: f64,
    fs: u32,
) -> f64 {
    if f0 < STONEMASK_F0_FLOOR || f0 > fs as f64 / 12.0 {
        return 0.0;
    }
    let half_window = (1.5 * fs as f64 / f0 + 1.0).floor() as usize;
    let window_time = (2.0 * half_window as f64 + 1.0) / fs as f64;
    let base_len = half_window * 2 + 1;
    let fftsize = (base_len * 2).next_power_of_two();

    let mean = mean_f0(
        planner,
        scratch,
        x,
        fs,
        tpos,
        f0,
        fftsize,
        window_time,
        base_len,
    );
    if (mean - f0).abs() / f0 > STONEMASK_THRESHOLD {
        f0
    } else {
        mean
    }
}

fn mean_f0(
    planner: &mut RealFftPlanner<f64>,
    scratch: &mut StoneMaskScratch,
    x: &[f64],
    fs: u32,
    tpos: f64,
    f0: f64,
    fftsize: usize,
    window_time: f64,
    window_length: usize,
) -> f64 {
    scratch.ensure_size(planner, fftsize, window_length);

    // 计算原始索引：round(tpos*fs) + (-half..half)
    let center = (tpos * fs as f64).round() as isize;
    for i in 0..window_length {
        scratch.index_raw[i] = center + (i as isize - (window_length as isize - 1) / 2);
    }

    // 窗函数与微分窗（复用缓冲）
    blackman_inplace(
        &scratch.index_raw,
        window_time,
        fs as f64,
        tpos,
        &mut scratch.main_window,
    );
    differential_inplace(&scratch.main_window, &mut scratch.diff_window);

    // 裁剪索引并写入 FFT 输入
    for i in 0..window_length {
        let idx = scratch.index_raw[i]
            .saturating_sub(1)
            .clamp(0, (x.len() - 1) as isize) as usize;
        scratch.indices[i] = idx;
        scratch.in_main[i] = x[idx] * scratch.main_window[i];
        scratch.in_diff[i] = x[idx] * scratch.diff_window[i];
    }
    // 剩余置零
    for i in window_length..fftsize {
        scratch.in_main[i] = 0.0;
        scratch.in_diff[i] = 0.0;
    }

    // 前向 FFT（避免借用冲突，先克隆 Arc 句柄）
    let r2c_plan = scratch.r2c().clone();
    r2c_plan
        .process(&mut scratch.in_main, &mut scratch.main_spec)
        .expect("r2c forward failed (main)");
    r2c_plan
        .process(&mut scratch.in_diff, &mut scratch.diff_spec)
        .expect("r2c forward failed (diff)");

    // 频域量：功率谱与 Im(conj(X)*dX)
    let half_bins = fftsize / 2 + 1;
    for k in 0..half_bins {
        let mx: Complex<f64> = scratch.main_spec[k];
        let dx: Complex<f64> = scratch.diff_spec[k];
        scratch.power_spec[k] = mx.norm_sqr();
        scratch.numerator_i[k] = (mx.conj() * dx).im;
    }

    // 两阶段瞬时频率校正
    instant_f0(
        &scratch.numerator_i[..half_bins],
        &scratch.power_spec[..half_bins],
        fs as f64,
        fftsize,
        f0,
    )
}

// blackman_into & differential_into moved to common::window

fn fix_f0(
    power_spectrum: &[f64],
    numerator_i: &[f64],
    fftsize: usize,
    fs: f64,
    initial_f0: f64,
    number_of_harmonics: usize,
) -> f64 {
    let nyquist_bin = fftsize / 2;
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for i in 0..number_of_harmonics {
        let harm = (i + 1) as f64;
        let mut index = (initial_f0 * fftsize as f64 / fs * harm).round() as isize;
        if index < 0 {
            index = 0;
        }
        if index as usize > nyquist_bin {
            index = nyquist_bin as isize;
        }
        let idx = index as usize;
        let p = power_spectrum[idx];
        let amp = p.sqrt();
        let inst_freq = if p == 0.0 {
            0.0
        } else {
            (idx as f64) * fs / fftsize as f64 + numerator_i[idx] / p * fs / (2.0 * PI)
        };
        numerator += amp * inst_freq;
        denominator += amp * harm;
    }
    const SAFEGUARD: f64 = 1e-12;
    numerator / (denominator + SAFEGUARD)
}

fn instant_f0(
    numerator: &[f64],
    powerspec: &[f64],
    fs: f64,
    fftsize: usize,
    initial_f0: f64,
) -> f64 {
    // 阶段一：2 个谐波粗估
    let tentative = fix_f0(powerspec, numerator, fftsize, fs, initial_f0, 2);
    if tentative <= 0.0 || tentative > initial_f0 * 2.0 {
        return 0.0;
    }
    // 阶段二：以粗估为中心，用 6 个谐波精修
    fix_f0(powerspec, numerator, fftsize, fs, tentative, 6)
}

// 复用缓冲的工作结构
struct StoneMaskScratch {
    fftsize: usize,
    window_len: usize,
    r2c: Option<Arc<dyn RealToComplex<f64>>>,
    in_main: Vec<f64>,
    in_diff: Vec<f64>,
    main_spec: Vec<Complex<f64>>,
    diff_spec: Vec<Complex<f64>>,
    index_raw: Vec<isize>,
    indices: Vec<usize>,
    main_window: Vec<f64>,
    diff_window: Vec<f64>,
    power_spec: Vec<f64>,
    numerator_i: Vec<f64>,
}

impl StoneMaskScratch {
    fn new() -> Self {
        Self {
            fftsize: 0,
            window_len: 0,
            r2c: None,
            in_main: Vec::new(),
            in_diff: Vec::new(),
            main_spec: Vec::new(),
            diff_spec: Vec::new(),
            index_raw: Vec::new(),
            indices: Vec::new(),
            main_window: Vec::new(),
            diff_window: Vec::new(),
            power_spec: Vec::new(),
            numerator_i: Vec::new(),
        }
    }

    fn ensure_size(
        &mut self,
        planner: &mut RealFftPlanner<f64>,
        fftsize: usize,
        window_len: usize,
    ) {
        if self.fftsize != fftsize {
            self.fftsize = fftsize;
            self.r2c = Some(planner.plan_fft_forward(fftsize));
            self.in_main.resize(fftsize, 0.0);
            self.in_diff.resize(fftsize, 0.0);
            self.main_spec = self.r2c.as_ref().unwrap().make_output_vec();
            self.diff_spec = self.r2c.as_ref().unwrap().make_output_vec();
            self.power_spec.resize(fftsize / 2 + 1, 0.0);
            self.numerator_i.resize(fftsize / 2 + 1, 0.0);
        }
        if self.window_len != window_len {
            self.window_len = window_len;
            self.index_raw.resize(window_len, 0);
            self.indices.resize(window_len, 0);
            self.main_window.resize(window_len, 0.0);
            self.diff_window.resize(window_len, 0.0);
        }
    }

    #[inline]
    fn r2c(&self) -> &Arc<dyn RealToComplex<f64>> {
        self.r2c.as_ref().expect("FFT plan not initialized")
    }
}
