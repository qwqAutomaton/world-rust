/// 聚合四类零交叉信息（negative/positive/peak/dip），以及各自计数。
pub struct ZeroCrossings {
    pub negative_interval_locations: Vec<f64>,
    pub negative_intervals: Vec<f64>,
    pub positive_interval_locations: Vec<f64>,
    pub positive_intervals: Vec<f64>,
    pub peak_interval_locations: Vec<f64>,
    pub peak_intervals: Vec<f64>,
    pub dip_interval_locations: Vec<f64>,
    pub dip_intervals: Vec<f64>,
}
/// 提取信号的四类零交叉信息(负向/正向/峰值/谷值)。
///
/// # 参数
/// - `filtered_signal`: 滤波后的信号
/// - `y_length`: 信号长度
/// - `actual_fs`: 采样率
///
/// # 返回值
/// 返回 ZeroCrossings 结构体,包含四类零交叉的位置和区间信息
pub fn get_four_zero_crossing_intervals(
    filtered_signal: &[f64],
    y_length: usize,
    actual_fs: f64,
) -> ZeroCrossings {
    if y_length == 0 {
        return ZeroCrossings {
            negative_interval_locations: Vec::new(),
            negative_intervals: Vec::new(),
            positive_interval_locations: Vec::new(),
            positive_intervals: Vec::new(),
            peak_interval_locations: Vec::new(),
            peak_intervals: Vec::new(),
            dip_interval_locations: Vec::new(),
            dip_intervals: Vec::new(),
        };
    }

    let buf = &filtered_signal[..y_length];

    // 负向零交叉
    let (neg_locs, neg_intervals) = zero_crossing_engine(buf, actual_fs);

    // 正向零交叉: 对取反信号操作
    let negated: Vec<f64> = buf.iter().map(|v| -v).collect();
    let (pos_locs, pos_intervals) = zero_crossing_engine(&negated, actual_fs);

    // 峰值: 离散导数的零交叉
    let diff: Vec<f64> = buf.windows(2).map(|w| w[0] - w[1]).collect();
    let (peak_locs, peak_intervals) = if diff.len() >= 2 {
        zero_crossing_engine(&diff, actual_fs)
    } else {
        (Vec::new(), Vec::new())
    };

    // 谷值: 负导数的零交叉
    let neg_diff: Vec<f64> = diff.iter().map(|v| -v).collect();
    let (dip_locs, dip_intervals) = if neg_diff.len() >= 2 {
        zero_crossing_engine(&neg_diff, actual_fs)
    } else {
        (Vec::new(), Vec::new())
    };

    ZeroCrossings {
        negative_interval_locations: neg_locs,
        negative_intervals: neg_intervals,
        positive_interval_locations: pos_locs,
        positive_intervals: pos_intervals,
        peak_interval_locations: peak_locs,
        peak_intervals: peak_intervals,
        dip_interval_locations: dip_locs,
        dip_intervals: dip_intervals,
    }
}
/// 检测信号的负向零交叉点并计算区间信息。
///
/// # 参数
/// - `filtered_signal`: 滤波后的信号
/// - `fs`: 采样率
///
/// # 返回值
/// 返回元组 (interval_locations, intervals),分别表示区间中心位置和对应的频率
fn zero_crossing_engine(filtered_signal: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    // 找到所有负向零交叉点的索引
    let edges: Vec<usize> = filtered_signal
        .windows(2)
        .enumerate()
        .filter_map(|(i, w)| {
            if w[0] > 0.0 && w[1] <= 0.0 {
                Some(i + 1)
            } else {
                None
            }
        })
        .collect();

    if edges.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    // 计算零交叉位置精确值 (线性插值)
    let fine_edges: Vec<f64> = edges
        .iter()
        .map(|&idx| {
            let ratio =
                filtered_signal[idx - 1] / (filtered_signal[idx - 1] - filtered_signal[idx]);
            (idx as f64) - ratio
        })
        .collect();

    // 计算区间中心位置和频率
    let (interval_locations, intervals): (Vec<f64>, Vec<f64>) = fine_edges
        .windows(2)
        .map(|w| {
            let center = (w[0] + w[1]) / 2.0 / fs;
            let freq = fs / (w[1] - w[0]);
            (center, freq)
        })
        .unzip();

    (interval_locations, intervals)
}
