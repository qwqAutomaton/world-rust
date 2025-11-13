use std::f64::consts::PI;


/// 就地填充 Nuttall 窗（系数与 `nuttall` 完全一致）。
#[inline]
pub fn nuttall_inplace(length: usize, out: &mut [f64]) {
    if length == 0 {
        return;
    }
    let len_minus_1 = (length - 1) as f64;
    for i in 0..length {
        let ratio = i as f64 / len_minus_1;
        let two_pi_ratio = 2.0 * PI * ratio;
        out[i] = 0.355768
            - 0.487396 * two_pi_ratio.cos()
            + 0.144232 * (2.0 * two_pi_ratio).cos()
            - 0.012604 * (3.0 * two_pi_ratio).cos();
    }
}

/// 就地填充 Blackman 型主窗（StoneMask 用）。
/// index_raw 使用与 WORLD 一致的时间对齐方式：tmp = (raw - 1)/fs - tpos。
#[inline]
pub fn blackman_inplace(index_raw: &[isize], window_time: f64, fs: f64, tpos: f64, out: &mut [f64]) {
    for (i, &raw) in index_raw.iter().enumerate() {
        let tmp = (raw as f64 - 1.0) / fs - tpos;
        out[i] = 0.42
            + 0.5 * (2.0 * PI * tmp / window_time).cos()
            + 0.08 * (4.0 * PI * tmp / window_time).cos();
    }
}

/// 就地生成微分窗（中心差分，符号与 WORLD 保持一致）。
#[inline]
pub fn differential_inplace(main_window: &[f64], diff: &mut [f64]) {
    let len = main_window.len();
    if len == 0 {
        return;
    }
    if len == 1 {
        diff[0] = 0.0;
        return;
    }
    diff[0] = -main_window[1] / 2.0;
    for i in 1..len - 1 {
        diff[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
    }
    diff[len - 1] = main_window[len - 2] / 2.0;
}
