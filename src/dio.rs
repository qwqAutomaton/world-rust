use crate::common::utils::{EPS, downsample, interp1};
use crate::common::window::nuttall_inplace;
use num_complex::Complex64;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::{f64::consts::PI, sync::Arc};

pub mod f0; // f0 contours
pub mod stonemask;
pub mod zc; // zero crossings // StoneMask optimization

pub const CUTOFF_FREQ: f64 = 50.0;
pub const SCORE_MAX: f64 = 100000.0;

#[derive(Clone, Copy, Debug)]
pub struct DioOption {
    pub channels_in_octave: f64,
    pub f0_ceil: f64,
    pub f0_floor: f64,
    pub frame_period: f64,
    pub speed: u32,
    pub allowed_range: f64,
}

impl Default for DioOption {
    fn default() -> Self {
        DioOption {
            channels_in_octave: 2.0,
            f0_ceil: 800.0,
            f0_floor: 71.0,
            frame_period: 5.0,
            speed: 1,
            allowed_range: 0.1,
        }
    }
}

pub struct F0Contour {
    pub f0: Vec<f64>,
    pub tpos: Vec<f64>,
}

/// 设计低切(高通)滤波器。
///
/// # 参数
/// - `n`: 滤波器有效长度
/// - `fft_size`: FFT 大小
///
/// # 返回值
/// 返回频域低切滤波器系数向量(长度为 fft_size)
fn design_low_cut_filter(n: usize, fft_size: usize) -> Vec<f64> {
    let mut filter: Vec<f64> = (1..=n)
        .map(|i| 0.5 * (1.0 - ((i as f64) * 2.0 * PI / (n + 1) as f64).cos()))
        .collect();

    let amplitude = -filter.iter().sum::<f64>();
    filter.iter_mut().for_each(|v| *v /= amplitude);

    // 扩展到 fft_size 并重新排列
    let half = (n - 1) / 2;
    let mut result = vec![0.0; fft_size];

    // 将滤波器系数循环移位并放置到结果中
    // 前半部分: filter[half..n] 放到 result[0..n-half]
    let front_len = n - half;
    result[0..front_len].copy_from_slice(&filter[half..n]);
    // 后半部分: filter[0..half] 放到 result[fft_size-half..fft_size]
    result[fft_size - half..fft_size].copy_from_slice(&filter[0..half]);

    result[0] += 1.0;
    result
}

/// 对信号进行 FFT 并应用低切滤波器。
///
/// # 参数
/// - `y`: 输入信号(可变引用,会被修改用于 FFT)
/// - `actual_fs`: 实际采样率
/// - `r2c`: 实数到复数的 FFT 计划器
///
/// # 返回值
/// 返回滤波后的频谱(复数向量)
fn get_spectrum_for_estimation(
    y: &mut Vec<f64>,
    actual_fs: f64,
    r2c: &Arc<dyn RealToComplex<f64>>,
) -> Vec<Complex64> {
    // 转频谱
    let mut yspec = r2c.make_output_vec();
    r2c.process(y, &mut yspec)
        .expect("r2c forward failed for original signal.");

    // 截止频率以采样点为单位
    let cutoff_in_sample = (actual_fs / CUTOFF_FREQ).round() as usize;
    // 构造低切滤波器
    let mut low_cut = design_low_cut_filter(cutoff_in_sample * 2 + 1, y.len());
    // 转频谱
    let mut fspec = r2c.make_output_vec();
    r2c.process(&mut low_cut, &mut fspec)
        .expect("r2c forward failed for filter.");

    // 卷积
    assert_eq!(yspec.len(), fspec.len());
    yspec.iter_mut().zip(fspec).for_each(|(s, f)| *s *= f);
    yspec
}

/// 从多个候选 f0 中选择每帧得分最低(最佳)的 f0。
///
/// # 参数
/// - `f0_length`: 帧数
/// - `f0_candidates`: 候选 f0 二维数组(每行是一个频带的所有帧)
/// - `f0_scores`: 对应的评分二维数组
///
/// # 返回值
/// 返回每帧最佳 f0 向量
fn get_best_f0_contour(
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    f0_scores: &[Vec<f64>],
) -> Vec<f64> {
    (0..f0_length)
        .map(|i| {
            let mut minpos = 0;
            for j in 1..f0_candidates.len() {
                if f0_scores[j][i] < f0_scores[minpos][i] {
                    minpos = j;
                }
            }
            f0_candidates[minpos][i]
        })
        .collect()
}

/// 对频谱应用低通滤波并返回时域信号。
///
/// # 参数
/// - `half_average_length`: 低通滤波器半宽
/// - `spec`: 输入频谱
/// - `fft_size`: FFT 大小
/// - `y_length`: 需要的输出长度
/// - `r2c`: 实数到复数的 FFT 计划器
/// - `c2r`: 复数到实数的 FFT 计划器
///
/// # 返回值
/// 返回滤波后的时域信号向量
fn get_filtered_signal<'a>(
    half_average_length: usize,
    spec: &[Complex64],
    fft_size: usize,
    y_length: usize,
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
    scratch: &'a mut DioFilterScratch,
) -> &'a [f64] {
    // 确保缓冲尺寸
    scratch.ensure_size(fft_size);

    // 生成长度为 half_average_length*4 的 Nuttall 窗，写入 lpf 前部，其余清零
    let win_len = half_average_length * 4;
    nuttall_inplace(win_len, &mut scratch.lpf[..win_len]);
    for v in &mut scratch.lpf[win_len..] {
        *v = 0.0;
    }

    // 窗的频谱
    r2c.process(&mut scratch.lpf, &mut scratch.fspec)
        .expect("r2c failed for Nuttall LPF.");

    // 卷积（逐点乘）
    let hb = fft_size / 2 + 1;
    for k in 0..hb {
        scratch.conv[k] = spec[k] * scratch.fspec[k];
    }

    // IFFT 回时域
    c2r.process(&mut scratch.conv, &mut scratch.filtered)
        .expect("c2r inverse failed");

    // 提取所需的输出片段（借用切片，避免分配）
    let start = half_average_length * 2;
    &scratch.filtered[start..start + y_length]
}

/// 从四组插值 f0 计算该频带的候选 f0 和评分。
///
/// # 参数
/// - `interpolated_f0_set`: 四组插值后的 f0 序列
/// - `f0_length`: 帧数
/// - `f0_floor`: f0 下限
/// - `f0_ceil`: f0 上限
/// - `boundary_f0`: 当前频带边界
///
/// # 返回值
/// 返回元组 (f0_candidate, f0_score)
fn get_f0_candidate_contour_sub(
    interpolated_f0_set: &[Vec<f64>],
    f0_length: usize,
    f0_floor: f64,
    f0_ceil: f64,
    boundary_f0: f64,
) -> (Vec<f64>, Vec<f64>) {
    (0..f0_length)
        .map(|i| {
            // 计算四组插值的平均值
            let avg = interpolated_f0_set.iter().map(|set| set[i]).sum::<f64>() / 4.0;

            // 计算标准差作为评分
            let variance: f64 = interpolated_f0_set
                .iter()
                .map(|set| {
                    let diff = set[i] - avg;
                    diff * diff
                })
                .sum::<f64>()
                / 3.0;

            let score = variance.sqrt();

            // 检查候选是否在合理范围内
            let is_valid =
                avg <= boundary_f0 && avg >= boundary_f0 / 2.0 && avg <= f0_ceil && avg >= f0_floor;

            if is_valid {
                (avg, score)
            } else {
                (0.0, SCORE_MAX)
            }
        })
        .unzip()
}

/// 基于零交叉信息计算该频带的 f0 候选和评分。
///
/// # 参数
/// - `zc`: 零交叉信息结构体
/// - `boundary_f0`: 频带边界
/// - `f0_floor`: f0 下限
/// - `f0_ceil`: f0 上限
/// - `temporal_positions`: 时间轴(帧中心时间)
/// - `f0_length`: 帧数
///
/// # 返回值
/// 返回元组 (f0_candidate, f0_score)
fn get_f0_candidate_contour(
    zc: &zc::ZeroCrossings,
    boundary_f0: f64,
    f0_floor: f64,
    f0_ceil: f64,
    temporal_positions: &[f64],
    f0_length: usize,
) -> (Vec<f64>, Vec<f64>) {
    if !((zc.negative_intervals.len() > 2)
        && (zc.positive_intervals.len() > 2)
        && (zc.peak_intervals.len() > 2)
        && (zc.dip_intervals.len() > 2))
    {
        return (vec![0f64; f0_length], vec![SCORE_MAX; f0_length]);
    }

    let interpolated_f0_set: Vec<Vec<f64>> = [
        interp1(
            &zc.negative_interval_locations,
            &zc.negative_intervals,
            temporal_positions,
        ),
        interp1(
            &zc.positive_interval_locations,
            &zc.positive_intervals,
            temporal_positions,
        ),
        interp1(
            &zc.peak_interval_locations,
            &zc.peak_intervals,
            temporal_positions,
        ),
        interp1(
            &zc.dip_interval_locations,
            &zc.dip_intervals,
            temporal_positions,
        ),
    ]
    .to_vec();
    get_f0_candidate_contour_sub(
        &interpolated_f0_set,
        f0_length,
        f0_floor,
        f0_ceil,
        boundary_f0,
    )
}

/// 从频谱中提取单个频带的 f0 候选和评分。
///
/// # 参数
/// - `boundary_f0`: 频带边界频率
/// - `fs`: 采样率
/// - `y_spectrum`: 输入频谱
/// - `y_length`: 时域信号长度
/// - `fft_size`: FFT 大小
/// - `f0_floor`: f0 下限
/// - `f0_ceil`: f0 上限
/// - `tpos`: 时间轴
/// - `r2c`: 实数到复数的 FFT 计划器
/// - `c2r`: 复数到实数的 FFT 计划器
///
/// # 返回值
/// 返回元组 (f0_candidate, f0_score)
fn get_f0_candidate_from_raw_event(
    boundary_f0: f64,
    fs: f64,
    y_spectrum: &[Complex64],
    y_length: usize,
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    tpos: &[f64],
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
    scratch: &mut DioFilterScratch,
) -> (Vec<f64>, Vec<f64>) {
    let half = (fs / boundary_f0 / 2.0).round() as usize;
    let filtered_signal =
        get_filtered_signal(half, y_spectrum, fft_size, y_length, &r2c, &c2r, scratch);

    let zc = zc::get_four_zero_crossing_intervals(&filtered_signal, y_length, fs);

    let (cand, score) =
        get_f0_candidate_contour(&zc, boundary_f0, f0_floor, f0_ceil, tpos, tpos.len());

    (cand, score)
}

/// 对所有频带计算 f0 候选和评分。
///
/// # 参数
/// - `bounds`: 所有频带的边界频率列表
/// - `actual_fs`: 实际采样率
/// - `y_length`: 时域信号长度
/// - `tpos`: 时间轴
/// - `y_spectrum`: 输入频谱
/// - `fft_size`: FFT 大小
/// - `f0_floor`: f0 下限
/// - `f0_ceil`: f0 上限
/// - `r2c`: 实数到复数的 FFT 计划器
/// - `c2r`: 复数到实数的 FFT 计划器
///
/// # 返回值
/// 返回元组 (raw_f0_candidates, raw_f0_scores),均为二维向量
fn get_f0_candidates_and_scores(
    bounds: &[f64],
    actual_fs: f64,
    y_length: usize,
    tpos: &[f64],
    y_spectrum: &[Complex64],
    fft_size: usize,
    f0_floor: f64,
    f0_ceil: f64,
    r2c: &Arc<dyn RealToComplex<f64>>,
    c2r: &Arc<dyn ComplexToReal<f64>>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut scratch = DioFilterScratch::default();
    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(bounds.len());
    let mut scores: Vec<Vec<f64>> = Vec::with_capacity(bounds.len());
    for &boundary in bounds.iter() {
        let (f0_candidate, f0_score) = get_f0_candidate_from_raw_event(
            boundary,
            actual_fs,
            y_spectrum,
            y_length,
            fft_size,
            f0_floor,
            f0_ceil,
            tpos,
            r2c,
            c2r,
            &mut scratch,
        );
        let normalized_score: Vec<f64> = f0_score
            .iter()
            .zip(f0_candidate.iter())
            .map(|(s, f)| s / (f + EPS))
            .collect();
        candidates.push(f0_candidate);
        scores.push(normalized_score);
    }
    (candidates, scores)
}

/// DIO 算法的核心实现。
///
/// # 参数
/// - `x`: 输入音频信号
/// - `fs`: 采样率(Hz)
/// - `frame_period`: 帧周期(毫秒)
/// - `f0_floor`: f0 下限
/// - `f0_ceil`: f0 上限
/// - `channels_in_octave`: 每个八度的频带数
/// - `speed`: 下采样速度(1-12)
/// - `allowed_range`: 允许的相对变化范围
///
/// # 返回值
/// 返回元组 (temporal_positions, f0),分别是时间轴和对应的 f0 值
fn dio_general_body(
    x: &[f64],
    fs: u32,
    frame_period: f64,
    f0_floor: f64,
    f0_ceil: f64,
    channels_in_octave: f64,
    speed: u32,
    allowed_range: f64,
) -> (Vec<f64>, Vec<f64>) {
    // 1. 计算频带边界(对数均分)
    let number_of_bands = 1 + ((f0_ceil / f0_floor).log2() * channels_in_octave) as usize;
    let boundary_f0_list: Vec<f64> = (0..number_of_bands)
        .map(|i| f0_floor * 2f64.powf((i as f64 + 1.0) / channels_in_octave))
        .collect();

    // 2. 下采样并去除直流分量
    let downsample_rate = speed.clamp(1, 12);
    let mut y = downsample(x, downsample_rate as usize);
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    y.iter_mut().for_each(|v| *v -= mean_y);
    let actual_fs = (fs as f64) / (downsample_rate as f64);

    // 3. FFT 预处理: 填充零以容纳低切滤波器和 Nuttall 窗
    let original_length = y.len();
    let lowcut_length = (actual_fs / CUTOFF_FREQ).round() as usize * 2 + 1;
    let nuttall_length = 4 * (1 + (actual_fs / boundary_f0_list[0] / 2.0) as usize);
    let padded_length = (original_length + lowcut_length + nuttall_length).next_power_of_two();
    y.resize(padded_length, 0.0);

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(y.len());
    let c2r = planner.plan_fft_inverse(y.len());
    let spec = get_spectrum_for_estimation(&mut y, actual_fs, &r2c);

    // 4. 计算时间轴
    let f0_length = (1000.0 * (x.len() as f64) / (fs as f64) / frame_period) as usize + 1;
    let temporal_positions: Vec<f64> = (0..f0_length)
        .map(|i| i as f64 * frame_period / 1000.0)
        .collect();

    // 5. 计算所有频带的候选 f0 和评分
    let (f0_candidates, f0_scores) = get_f0_candidates_and_scores(
        &boundary_f0_list,
        actual_fs,
        original_length,
        &temporal_positions,
        &spec,
        y.len(),
        f0_floor,
        f0_ceil,
        &r2c,
        &c2r,
    );

    // 6. 选择每帧最佳 f0
    let best_f0_contour = get_best_f0_contour(f0_length, &f0_candidates, &f0_scores);

    // 7. 后处理平滑
    let fixed = f0::fix_f0_contour(
        frame_period,
        &f0_candidates,
        &best_f0_contour,
        f0_length,
        f0_floor,
        allowed_range,
    );

    (temporal_positions, fixed)
}

// 就地 Nuttall 窗填充，系数与 common::window::nuttall 保持一致
// fill_nuttall_into 已移动到 common::window 模块

// DIO 低通卷积阶段复用缓冲
#[derive(Default)]
struct DioFilterScratch {
    lpf: Vec<f64>,
    fspec: Vec<Complex64>,
    conv: Vec<Complex64>,
    filtered: Vec<f64>,
}

impl DioFilterScratch {
    fn ensure_size(&mut self, fft_size: usize) {
        if self.lpf.len() != fft_size {
            self.lpf.resize(fft_size, 0.0);
        }
        let hb = fft_size / 2 + 1;
        if self.fspec.len() != hb {
            self.fspec.resize(hb, Complex64::new(0.0, 0.0));
        }
        if self.conv.len() != hb {
            self.conv.resize(hb, Complex64::new(0.0, 0.0));
        }
        if self.filtered.len() != fft_size {
            self.filtered.resize(fft_size, 0.0);
        }
    }
}

/// 执行 DIO (Distributed Inline-filter Operation) 基频估计算法。
///
/// # 参数
/// - `x`: 输入音频信号
/// - `fs`: 采样率(Hz)
/// - `option`: DIO 算法配置选项
///
/// # 返回值
/// 返回 F0Contour 结构体,包含时间轴和对应的 f0 值
pub fn dio(x: &[f64], fs: u32, option: &DioOption) -> F0Contour {
    let start = std::time::Instant::now();
    let (tpos, f0) = dio_general_body(
        x,
        fs,
        option.frame_period,
        option.f0_floor,
        option.f0_ceil,
        option.channels_in_octave,
        option.speed,
        option.allowed_range,
    );
    eprintln!("DIO done. Time: {:?}", start.elapsed());
    let start = std::time::Instant::now();
    let refined = stonemask::refine_f0_contour(x, &f0, &tpos, fs);
    eprintln!("StoneMask done. Time: {:?}", start.elapsed());
    F0Contour { f0: refined, tpos }
}
