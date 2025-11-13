use crate::common::utils::EPS;

// 旧的分配版 fix_step1 已移除；见 fix_step1_into
// 旧的分配版 fix_step2 已移除；见 fix_step2_into
// 旧的分配版 fix_step3 已移除；见 fix_step3_inplace
// 旧的分配版 fix_step4 已移除；见 fix_step4_inplace
/// 完整的 f0 修正流程,组合多个步骤得到最终平滑的 f0 轮廓。
///
/// # 参数
/// - `frame_period`: 帧周期(毫秒)
/// - `f0_candidates`: 所有频带的候选 f0
/// - `best_f0_contour`: 初始最佳 f0
/// - `f0_length`: 帧数
/// - `f0_floor`: f0 下限
/// - `allowed_range`: 允许的相对变化范围
/// # 返回值
/// 返回修正后的 f0 向量（使用 FixF0Scratch）。
pub fn fix_f0_contour(
    frame_period: f64,
    f0_candidates: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
) -> Vec<f64> {
    let mut scratch = FixF0Scratch::default();
    fix_f0_contour_with_scratch(
        frame_period,
        f0_candidates,
        best_f0_contour,
        f0_length,
        f0_floor,
        allowed_range,
        &mut scratch,
    )
}

/// 与 fix_f0_contour 等价，但允许复用内部缓冲减少分配。
pub fn fix_f0_contour_with_scratch(
    frame_period: f64,
    f0_candidates: &[Vec<f64>],
    best_f0_contour: &[f64],
    f0_length: usize,
    f0_floor: f64,
    allowed_range: f64,
    scratch: &mut FixF0Scratch,
) -> Vec<f64> {
    let voice_range_minimum =
        ((1000.0 / frame_period / f0_floor).round() as usize * 2 + 1) as usize;
    if f0_length <= voice_range_minimum {
        return vec![0.0_f64; f0_length];
    }

    scratch.ensure_len(f0_length);

    // step1 -> tmp
    // 先按原算法生成一步结果（需要 into 版本改造时可替换）
    fix_step1_into(
        best_f0_contour,
        voice_range_minimum,
        allowed_range,
        &mut scratch.tmp,
    );

    // step2 -> out（线性时间）
    let mut out = vec![0.0_f64; f0_length];
    fix_step2_into(&scratch.tmp, f0_length, voice_range_minimum, &mut out);

    // 浊音段边界
    compute_voiced_sections(
        &out,
        f0_length,
        &mut scratch.positive_index,
        &mut scratch.negative_index,
    );

    // step3 forward（就地）
    fix_step3_inplace(
        &mut out,
        f0_candidates,
        allowed_range,
        &scratch.negative_index,
    );

    // step4 backward（就地）
    fix_step4_inplace(
        &mut out,
        f0_candidates,
        allowed_range,
        &scratch.positive_index,
    );

    out
}

// 内部：就地/into 变体
fn fix_step1_into(f0: &[f64], voice_range_minimum: usize, allowed_range: f64, out: &mut [f64]) {
    let len = f0.len();
    debug_assert_eq!(out.len(), len);

    for i in 0..len {
        if i < voice_range_minimum || i >= len.saturating_sub(voice_range_minimum) {
            out[i] = 0.0;
            continue;
        }

        let cur = f0[i];
        let prev =
            if i >= 1 && (i - 1) >= voice_range_minimum && (i - 1) < len - voice_range_minimum {
                f0[i - 1]
            } else {
                0.0
            };
        let change_ratio = (cur - prev).abs() / (cur + EPS);
        out[i] = if change_ratio < allowed_range {
            cur
        } else {
            0.0
        };
    }
}

/// Step2: 小窗口直接扫描（O(n*w)），w 很小时常数更优。
fn fix_step2_into(f0: &[f64], f0_length: usize, voice_range_minimum: usize, out: &mut [f64]) {
    let center = (voice_range_minimum - 1) / 2;
    for i in 0..f0_length {
        if i < center || i >= f0_length - center {
            out[i] = f0[i];
        } else {
            let l = i - center;
            let r = i + center;
            let mut has_zero = false;
            for j in l..=r {
                if f0[j].abs() < EPS {
                    has_zero = true;
                    break;
                }
            }
            out[i] = if has_zero { 0.0 } else { f0[i] };
        }
    }
}

fn compute_voiced_sections(
    f0: &[f64],
    f0_length: usize,
    positive_index: &mut Vec<usize>,
    negative_index: &mut Vec<usize>,
) {
    positive_index.clear();
    negative_index.clear();
    for i in 1..f0_length {
        match (f0[i - 1] == 0.0, f0[i] == 0.0) {
            (true, false) => positive_index.push(i),     // 从清音到浊音
            (false, true) => negative_index.push(i - 1), // 从浊音到清音
            _ => {}
        }
    }
}

fn fix_step3_inplace(
    out: &mut [f64],
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    negative_index: &[usize],
) {
    let f0_length = out.len();
    for (i, &neg_idx) in negative_index.iter().enumerate() {
        let limit = if i == negative_index.len() - 1 {
            f0_length - 1
        } else {
            negative_index[i + 1]
        };
        for j in neg_idx..limit {
            let prev = if j >= 1 { out[j - 1] } else { 0.0 };
            out[j + 1] = select_best_f0(out[j], prev, f0_candidates, j + 1, allowed_range);
            if out[j + 1] == 0.0 {
                break;
            }
        }
    }
}

fn fix_step4_inplace(
    out: &mut [f64],
    f0_candidates: &[Vec<f64>],
    allowed_range: f64,
    positive_index: &[usize],
) {
    if positive_index.is_empty() {
        return;
    }

    for i in (0..positive_index.len()).rev() {
        let idx = positive_index[i];
        let limit = if i == 0 { 1 } else { positive_index[i - 1] };

        for j in (limit..idx).rev() {
            let next = out.get(j + 1).copied().unwrap_or(0.0);
            out[j] = select_best_f0(out[j + 1], next, f0_candidates, j, allowed_range);
            if out[j] == 0.0 {
                break;
            }
        }
    }
}
/// 检测浊音段的起止边界。
///
/// # 参数
/// - `f0`: f0 序列
/// - `f0_length`: 帧数
///
/// # 返回值
/// 返回元组 (positive_index, negative_index),分别表示浊音段的起始和结束帧索引
// 旧的返回元组版 get_number_of_voiced_sections 已移除；见 compute_voiced_sections
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

/// DIO 修正阶段可复用的缓冲区，减少临时分配。
#[derive(Default)]
pub struct FixF0Scratch {
    // step1 输出暂存
    tmp: Vec<f64>,
    // 浊音段边界缓存
    positive_index: Vec<usize>,
    negative_index: Vec<usize>,
}

impl FixF0Scratch {
    fn ensure_len(&mut self, n: usize) {
        if self.tmp.len() != n {
            self.tmp.resize(n, 0.0);
        }
    }
}
