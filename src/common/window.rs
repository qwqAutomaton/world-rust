use std::f64::consts::PI;

/// 生成 Nuttall 窗函数。
///
/// # 参数
/// - `length`: 窗函数长度
///
/// # 返回值
/// 返回 Nuttall 窗函数系数向量
pub fn get_nuttall_window(length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }

    let len_minus_1 = (length - 1) as f64;
    (0..length)
        .map(|i| {
            let ratio = i as f64 / len_minus_1;
            let two_pi_ratio = 2.0 * PI * ratio;
            0.355768 - 0.487396 * two_pi_ratio.cos() + 0.144232 * (2.0 * two_pi_ratio).cos()
                - 0.012604 * (3.0 * two_pi_ratio).cos()
        })
        .collect()
}
