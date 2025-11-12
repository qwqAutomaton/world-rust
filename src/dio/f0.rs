/// Fix 步骤 1: 过滤短时异常跳变,生成初步平滑的 f0。
///
/// # 参数
/// - `f0`: 输入 f0 序列
/// - `voice_range_minimum`: 最小浊音段长度
/// - `allowed_range`: 允许的相对变化范围
///
/// # 返回值
/// 返回平滑后的 f0 向量
fn fix_step1(f0: &[f64], voice_range_minimum: usize, allowed_range: f64) -> Vec<f64> {
    let len = f0.len();

    // 构建 f0_base: 去掉首尾的 voice_range_minimum 帧
    let f0_base: Vec<f64> = if len > voice_range_minimum * 2 {
        (0..len)
            .map(|i| {
                if i >= voice_range_minimum && i < len - voice_range_minimum {
                    f0[i]
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![0.0; len]
    };

    // 检查相邻帧的跳变
    (0..len)
        .map(|i| {
            if i < voice_range_minimum {
                0.0
            } else {
                let change_ratio = (f0_base[i] - f0_base[i - 1]).abs() / (f0_base[i] + super::EPS);
                if change_ratio < allowed_range {
                    f0_base[i]
                } else {
                    0.0
                }
            }
        })
        .collect()
}
/// Fix 步骤 2: 局部一致性检查,去除孤立的浊音帧。
///
/// # 参数
/// - `f0_step1`: 步骤 1 的输出
/// - `f0_length`: 帧数
/// - `voice_range_minimum`: 最小浊音段长度
///
/// # 返回值
/// 返回检查后的 f0 向量
fn fix_step2(f0_step1: &[f64], f0_length: usize, voice_range_minimum: usize) -> Vec<f64> {
    let center = (voice_range_minimum - 1) / 2;

    (0..f0_length)
        .map(|i| {
            if i < center || i >= f0_length - center {
                f0_step1[i]
            } else {
                // 检查窗口内是否有任何0值
                let has_zero = (i - center..=i + center).any(|j| f0_step1[j] == 0.0);
                if has_zero { 0.0 } else { f0_step1[i] }
            }
        })
        .collect()
}
/// Fix 步骤 3: 向前扩展浊音段,填补浊音段末尾的缺失 f0。
///
/// # 参数
/// - `f0_step2`: 步骤 2 的输出
/// - `f0_length`: 帧数
/// - `f0_candidates`: 所有频带的候选 f0
/// - `allowed_range`: 允许的相对变化范围
/// - `negative_index`: 浊音段结束帧索引列表
///
/// # 返回值
/// 返回向前扩展后的 f0 向量
fn fix_step3(
    f0_step2: &[f64],
    f0_length: usize,
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    negative_index: &[usize],
) -> Vec<f64> {
    let mut f0_step3 = vec![0.0_f64; f0_length];
    f0_step3.copy_from_slice(f0_step2);
    for (i, &neg_idx) in negative_index.iter().enumerate() {
        let limit = if i == negative_index.len() - 1 {
            f0_length - 1
        } else {
            negative_index[i + 1]
        };
        for j in neg_idx..limit {
            let prev = if j >= 1 { f0_step3[j - 1] } else { 0.0 };
            f0_step3[j + 1] =
                select_best_f0(f0_step3[j], prev, f0_candidates, j + 1, allowed_range);
            if f0_step3[j + 1] == 0.0 {
                break;
            }
        }
    }

    f0_step3
}
/// Fix 步骤 4: 向后扩展浊音段,填补浊音段开头的缺失 f0。
///
/// # 参数
/// - `f0_step3`: 步骤 3 的输出
/// - `f0_candidates`: 所有频带的候选 f0
/// - `allowed_range`: 允许的相对变化范围
/// - `positive_index`: 浊音段起始帧索引列表
///
/// # 返回值
/// 返回向后扩展后的 f0 向量
fn fix_step4(
    f0_step3: &[f64],
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    positive_index: &[usize],
) -> Vec<f64> {
    let mut f0_step4 = f0_step3.to_vec();

    if positive_index.is_empty() {
        return f0_step4;
    }

    for i in (0..positive_index.len()).rev() {
        let idx = positive_index[i];
        let limit = if i == 0 { 1 } else { positive_index[i - 1] };

        for j in (limit..idx).rev() {
            let next = f0_step4.get(j + 1).copied().unwrap_or(0.0);
            f0_step4[j] = select_best_f0(f0_step4[j + 1], next, f0_candidates, j, allowed_range);
            if f0_step4[j] == 0.0 {
                break;
            }
        }
    }

    f0_step4
}
/// 完整的 f0 修正流程,组合多个步骤得到最终平滑的 f0 轮廓。
///
/// # 参数
/// - `frame_period`: 帧周期(毫秒)
/// - `f0_candidates`: 所有频带的候选 f0
/// - `best_f0_contour`: 初始最佳 f0
/// - `f0_length`: 帧数
/// - `f0_floor`: f0 下限
/// - `allowed_range`: 允许的相对变化范围
///
/// # 返回值
/// 返回修正后的 f0 向量
pub fn fix_f0_contour(
    frame_period: f64,
    f0_candidates: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
) -> Vec<f64> {
    let voice_range_minimum = ((0.5 + 1000.0 / frame_period / f0_floor) as i32 * 2 + 1) as usize;
    if f0_length <= voice_range_minimum {
        return vec![0.0_f64; f0_length];
    }

    // 排除跳变，跳变了赋 0
    let f0_tmp1 = fix_step1(best_f0_contour, voice_range_minimum, allowed_range);

    // 给上面赋 0 的帧做插值补上
    let f0_tmp2 = fix_step2(&f0_tmp1, f0_length, voice_range_minimum);

    // 检测浊音段
    let (positive_index, negative_index) = get_number_of_voiced_sections(&f0_tmp2, f0_length);

    let f0_tmp1 = fix_step3(
        &f0_tmp2,
        f0_length,
        f0_candidates,
        allowed_range,
        &negative_index,
    );
    let fixed_f0_contour = fix_step4(&f0_tmp1, f0_candidates, allowed_range, &positive_index);

    fixed_f0_contour
}
/// 检测浊音段的起止边界。
///
/// # 参数
/// - `f0`: f0 序列
/// - `f0_length`: 帧数
///
/// # 返回值
/// 返回元组 (positive_index, negative_index),分别表示浊音段的起始和结束帧索引
fn get_number_of_voiced_sections(f0: &[f64], f0_length: usize) -> (Vec<usize>, Vec<usize>) {
    let mut positive_index = Vec::new();
    let mut negative_index = Vec::new();

    for i in 1..f0_length {
        match (f0[i - 1] == 0.0, f0[i] == 0.0) {
            (true, false) => positive_index.push(i),     // 从清音到浊音
            (false, true) => negative_index.push(i - 1), // 从浊音到清音
            _ => {}
        }
    }

    (positive_index, negative_index)
}
/// 基于参考值从候选 f0 中选择最优 f0。
///
/// # 参数
/// - `current_f0`: 当前帧 f0
/// - `past_f0`: 前一帧 f0
/// - `f0_candidates`: 所有频带的候选 f0
/// - `target_index`: 目标帧索引
/// - `allowed_range`: 允许的相对变化范围
///
/// # 返回值
/// 返回选择的最优 f0,如果超出范围则返回 0.0
fn select_best_f0(
    current_f0: f64,
    past_f0: f64,
    f0_candidates: &[Vec<f64>],
    target_index: usize,
    allowed_range: f64,
) -> f64 {
    let reference_f0 = (current_f0 * 3.0 - past_f0) / 2.0;

    let best_f0 = f0_candidates
        .iter()
        .map(|band| band[target_index])
        .min_by(|&a, &b| {
            let error_a = (reference_f0 - a).abs();
            let error_b = (reference_f0 - b).abs();
            error_a.partial_cmp(&error_b).unwrap()
        })
        .unwrap_or(0.0);

    if (1.0 - best_f0 / reference_f0).abs() > allowed_range {
        0.0
    } else {
        best_f0
    }
}
