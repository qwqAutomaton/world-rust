pub const EPS: f64 = 1e-10;

/// 对信号进行带抗混叠滤波的下采样。
///
/// # 参数
/// - `x`: 输入信号切片
/// - `r`: 下采样比率
///
/// # 返回值
/// 返回下采样后的信号向量
pub fn downsample(x: &[f64], r: usize) -> Vec<f64> {
    // 不下采样，原样返回
    if r == 1 {
        return x.to_vec();
    }
    // 镜像延展，抗混叠
    const K_NFACT: usize = 9;
    let mut tmp1 = (0..K_NFACT)
        .map(|i| 2.0 * x[0] - x[K_NFACT - i]) // 左边
        .chain(x.iter().copied()) // 原信号
        .chain((0..K_NFACT).map(|i| 2.0 * x[x.len() - 1] - x[x.len() - 2 - i])) // 右边
        .collect::<Vec<f64>>();

    // 双向滤波
    let mut tmp2 = filter_for_decimate(&tmp1, r);
    assert_eq!(tmp1.len(), tmp2.len());
    tmp2.reverse();
    tmp1 = filter_for_decimate(&tmp2, r);
    tmp1.reverse();

    // 下采样
    let end = x.len() + K_NFACT - 1; // 右边界
    let cnt = (x.len() - 1) / r + 1; // 下采样后长度
    let beg = end - r * (cnt - 1); // 左边界

    (beg..(tmp1.len()))
        .step_by(r)
        .take(cnt)
        .map(|i| tmp1[i])
        .collect::<Vec<f64>>()
}
/// 对输入信号应用 IIR 低通滤波器,用于下采样前的抗混叠处理。
///
/// # 参数
/// - `x`: 输入信号切片
/// - `r`: 下采样比率(2-12),决定使用哪组滤波器系数
///
/// # 返回值
/// 返回滤波后的信号向量
fn filter_for_decimate(x: &[f64], r: usize) -> Vec<f64> {
    // 使用元组直接返回系数,避免可变赋值
    let (a, b) = match r {
        11 => (
            [2.450743295230728, -2.06794904601978, 0.59574774438332101],
            [0.0026822508007163792, 0.0080467524021491377],
        ),
        12 => (
            [2.4981398605924205, -2.1368928194784025, 0.62187513816221485],
            [0.0021097275904709001, 0.0063291827714127002],
        ),
        10 => (
            [2.3936475118069387, -1.9873904075111861, 0.5658879979027055],
            [0.0034818622251927556, 0.010445586675578267],
        ),
        9 => (
            [2.3236003491759578, -1.8921545617463598, 0.53148928133729068],
            [0.0046331164041389372, 0.013899349212416812],
        ),
        8 => (
            [2.2357462340187593, -1.7780899984041358, 0.49152555365968692],
            [0.0063522763407111993, 0.019056829022133598],
        ),
        7 => (
            [2.1225239019534703, -1.6395144861046302, 0.44469707800587366],
            [0.0090366882681608418, 0.027110064804482525],
        ),
        6 => (
            [1.9715352749512141, -1.4686795689225347, 0.3893908434965701],
            [0.013469181309343825, 0.040407543928031475],
        ),
        5 => (
            [1.7610939654280557, -1.2554914843859768, 0.3237186507788215],
            [0.021334858522387423, 0.06400457556716227],
        ),
        4 => (
            [
                1.4499664446880227,
                -0.98943497080950582,
                0.24578252340690215,
            ],
            [0.036710750339322612, 0.11013225101796784],
        ),
        3 => (
            [
                0.95039378983237421,
                -0.67429146741526791,
                0.15412211621346475,
            ],
            [0.071221945171178636, 0.21366583551353591],
        ),
        2 => (
            [
                0.041156734567757189,
                -0.42599112459189636,
                0.041037215479961225,
            ],
            [0.16797464681802227, 0.50392394045406674],
        ),
        _ => ([0.0, 0.0, 0.0], [0.0, 0.0]),
    };

    let mut w = [0.0; 3];
    x.iter()
        .map(|&xi| {
            let wt = xi + a[0] * w[0] + a[1] * w[1] + a[2] * w[2];
            let yi = b[0] * wt + b[1] * w[0] + b[1] * w[1] + b[0] * w[2];
            w[2] = w[1];
            w[1] = w[0];
            w[0] = wt;
            yi
        })
        .collect()
}
#[allow(dead_code)]
/// 对给定的采样点进行线性插值。
///
/// # 参数
/// - `xp`: 已知数据点的 x 坐标(单调递增)
/// - `fp`: 已知数据点的 y 坐标
/// - `x`: 需要插值的 x 坐标切片
///
/// # 返回值
/// 返回插值后的 y 坐标向量
pub fn interp1(xp: &[f64], fp: &[f64], x: &[f64]) -> Vec<f64> {
    if xp.len() < 2 {
        return vec![0.0; x.len()];
    }

    x.iter()
        .map(|&xi| {
            // 边界情况
            if xi <= xp[0] {
                return fp[0];
            }
            if xi >= xp[xp.len() - 1] {
                return fp[xp.len() - 1];
            }

            // 二分查找左边界
            let idx = match xp.binary_search_by(|probe| probe.partial_cmp(&xi).unwrap()) {
                Ok(i) => i,
                Err(i) => i - 1,
            };

            // 线性插值
            let t = (xi - xp[idx]) / (xp[idx + 1] - xp[idx]);
            fp[idx] + t * (fp[idx + 1] - fp[idx])
        })
        .collect()
}

/// interp1q: 快速线性插值，仿照 WORLD C++ 版本
/// x: 起点（等间隔），shift: 间隔，y: 原始数据，xi: 目标点
pub fn interp1q(x: f64, shift: f64, y: &[f64], xi: &[f64]) -> Vec<f64> {
    let x_length = y.len();
    let xi_length = xi.len();
    let mut yi = vec![0.0; xi_length];
    let mut delta_y = vec![0.0; x_length];
    for i in 0..x_length - 1 {
        delta_y[i] = y[i + 1] - y[i];
    }
    delta_y[x_length - 1] = 0.0;

    // ...existing code...
    for i in 0..xi_length {
        let base = ((xi[i] - x) / shift).floor() as isize;
        let frac = (xi[i] - x) / shift - base as f64;
        let base = base.clamp(0, (x_length - 1) as isize) as usize;
        yi[i] = y[base] + delta_y[base] * frac;
    }
    yi
}

pub fn linear_smoothing(input: &[f64], width: f64, fs: u32, fft_size: usize) -> Vec<f64> {
    let boundary = (width * fft_size as f64 / fs as f64).round() as usize + 1;
    let fft_size_half = fft_size / 2;

    let mut mirroring_spectrum = vec![0.0; fft_size_half + 2 * boundary + 1];
    let mut mirroring_segment = vec![0.0; fft_size_half + 2 * boundary + 1];
    let mut frequency_axis = vec![0.0; fft_size_half + 1];

    //
    for i in 0..boundary {
        mirroring_spectrum[i] = input[boundary - i];
    }
    for i in 0..=fft_size_half {
        mirroring_spectrum[i + boundary] = input[i];
    }
    for i in 0..boundary {
        mirroring_spectrum[fft_size_half + boundary + 1 + i] = input[fft_size_half - 1 - i];
    }

    mirroring_segment[0] = mirroring_spectrum[0] * fs as f64 / fft_size as f64;
    for i in 1..mirroring_segment.len() {
        mirroring_segment[i] =
            mirroring_spectrum[i] * fs as f64 / fft_size as f64 + mirroring_segment[i - 1];
    }

    for i in 0..=fft_size_half {
        frequency_axis[i] = i as f64 / fft_size as f64 * fs as f64 - width / 2.0;
    }

    let origin_of_mirroring_axis = -(boundary as f64 - 0.5) * fs as f64 / fft_size as f64;
    let discrete_frequency_interval = fs as f64 / fft_size as f64;

    let low_levels = interp1q(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        &mirroring_segment,
        &frequency_axis,
    );

    for i in 0..=fft_size_half {
        frequency_axis[i] += width;
    }

    let high_levels = interp1q(
        origin_of_mirroring_axis,
        discrete_frequency_interval,
        &mirroring_segment,
        &frequency_axis,
    );

    let mut output = vec![0.0; fft_size_half + 1];
    for i in 0..=fft_size_half {
        output[i] = (high_levels[i] - low_levels[i]) / width;
    }
    output
}

pub fn dc_fix(input: &[f64], f0: f64, fs: u32, fft_size: usize) -> Vec<f64> {
    let mut output = input.to_vec();
    let upper_limit = (2.0 + f0 * fft_size as f64 / fs as f64) as usize;
    if upper_limit >= input.len() {
        return output;
    }

    let low_frequency_axis: Vec<f64> = (0..upper_limit)
        .map(|i| i as f64 * fs as f64 / fft_size as f64)
        .collect();

    let upper_limit_replica = upper_limit;

    // C++ interp1Q in this case is equivalent to interp1 with reversed x-axis
    let replica_x_axis: Vec<f64> = low_frequency_axis.iter().map(|&x| f0 - x).collect();
    let input_x_axis: Vec<f64> = (0..input.len())
        .map(|i| i as f64 * fs as f64 / fft_size as f64)
        .collect();

    let low_frequency_replica = interp1(&input_x_axis, input, &replica_x_axis);

    for i in 0..upper_limit_replica {
        output[i] += low_frequency_replica[i];
    }

    output
}
