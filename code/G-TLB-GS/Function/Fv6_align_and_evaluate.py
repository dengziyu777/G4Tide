import numpy as np
import os

from source.Fv6_align_sequences import Fv6_align_sequences
from source.Fv6_DebugTools import debug_print_meteo_data, debug_plot_interpolated_scatter


def Fv6_align_and_evaluate(fc_overlap_list, obs_overlap_list, forward_hours, time_interval,
                           output_evaluation_path, output_debug_folder=None, debug_mode=False, print_on = True):
    """
    对每个站点的预报和观测数据进行时间对齐、质量评估和可视化

    参数:
        fc_overlap_list (list): 预报数据列表，每个元素是一个站点的数据数组 [时间戳, 值]
        obs_overlap_list (list): 观测数据列表，每个元素是一个站点的数据数组 [时间戳, 值]
        forward_hours (int): 预测提前小时数
        time_interval (int): 观测数据时间间隔(秒)
        output_evaluation_path (str): 评估结果输出文件路径
        output_debug_folder (str): 调试图像输出目录
        debug_mode (bool): 是否生成调试图像
        print_on：执行函数时，是否打印信息

    返回:
        tuple: (
            X_full_list,      # 完整时间段的预测值列表
            t_full_list,       # 完整时间段的时间戳列表
            X_common_list,    # 公共时间段的预测值列表
            Y_common_list,    # 公共时间段的观测值列表
            t_common_list      # 公共时间段的时间戳列表
        )
    """
    # 初始化结果列表
    X_full_list = []
    t_full_list = []
    X_common_list = []
    Y_common_list = []
    t_common_list = []

    # 确保输出目录存在
    if debug_mode and output_debug_folder:
        os.makedirs(output_debug_folder, exist_ok=True)

    # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
    #     # 写入评估标题
    #     f.write(f"\n{'=' * 80}\n")
    #     f.write("潮位数据对齐评估结果\n")
    #     f.write(f"预测提前量: {forward_hours}小时 | 时间间隔: {time_interval}秒\n")
    #     f.write("站点\t预测点\t观测点\t对齐预测点(完整)\t对齐观测点(公共)\t有效数据点\n")

    # 对每个站点单独处理
    total_sites = min(len(fc_overlap_list), len(obs_overlap_list))
    for site_idx in range(total_sites):
        # 获取当前站点的预报和观测数据
        fc_overlap = fc_overlap_list[site_idx]
        obs_overlap = obs_overlap_list[site_idx]

        # # 更新评估文件
        # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
        #     f.write(f"{site_idx + 1}\t{len(fc_overlap) if len(fc_overlap) > 0 else 0}\t"
        #             f"{len(obs_overlap) if len(obs_overlap) > 0 else 0}\t")

        # 检查数据是否为空
        if len(fc_overlap) == 0 or len(obs_overlap) == 0:
            print(f"站点 {site_idx + 1}: 无可用数据，跳过对齐")
            # 添加空数组以保持列表长度一致
            X_full_list.append(np.array([]))
            t_full_list.append(np.array([]))
            X_common_list.append(np.array([]))
            Y_common_list.append(np.array([]))
            t_common_list.append(np.array([]))

            # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
            #     f.write("0\t0\t0\n")
            continue

        # 对齐当前站点的数据
        try:
            forecast_interp, t_target, obs_interp, t_target_orig = Fv6_align_sequences(
                fc_overlap[:, 0], fc_overlap[:, 1],
                forward_hours,
                obs_overlap[:, 0], obs_overlap[:, 1],
                time_interval
            )

            # 检查对齐结果
            if len(forecast_interp) == 0:
                print(f"站点 {site_idx + 1}: 对齐失败")
                X_full_list.append(np.array([]))
                t_full_list.append(np.array([]))
                X_common_list.append(np.array([]))
                Y_common_list.append(np.array([]))
                t_common_list.append(np.array([]))

                # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
                #     f.write("0\t0\t0\n")
                continue
        except Exception as e:
            print(f"站点 {site_idx + 1}: 对齐过程中出错 - {e}")
            X_full_list.append(np.array([]))
            t_full_list.append(np.array([]))
            X_common_list.append(np.array([]))
            Y_common_list.append(np.array([]))
            t_common_list.append(np.array([]))

            # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
            #     f.write("0\t0\t0\n")
            continue

        # 提取公共时间段的数据（去掉预报数据的提前部分）
        num_forward_steps = len(t_target) - len(t_target_orig)
        if num_forward_steps < 0:
            num_forward_steps = 0  # 防止负值索引错误

        # 保存完整时间段的数据（用于后续训练）
        X_full = forecast_interp
        t_full = t_target

        # 保存公共时间段的数据（用于评估）
        X_common = forecast_interp[num_forward_steps:]
        Y_common = obs_interp
        t_common = t_target_orig

        # 添加到结果列表
        X_full_list.append(X_full)
        t_full_list.append(t_full)
        X_common_list.append(X_common)
        Y_common_list.append(Y_common)
        t_common_list.append(t_common)

        # 写入对齐结果统计
        # with open(output_evaluation_path, 'a', encoding='utf-8') as f:
        #     f.write(f"{len(X_full)}\t{len(Y_common)}\t{len(t_common)}\n")

        # 绘制散点图（使用公共时间段数据）
        if debug_mode and output_debug_folder:
            debug_plot_interpolated_scatter(
                t_common, X_common, Y_common,
                save_path=os.path.join(output_debug_folder, f'site_{site_idx + 1}_scatter.png'),
                station_idx=site_idx
            )
        if print_on:
            print(f"站点{site_idx + 1}完成 // 插值后预测数据完整点（考虑历史时间步长）: {len(X_full)}, 公共点（不考虑历史时间步长）: {len(X_common)}")

    print(f"数据对齐评估完成，共{total_sites}个站点处理")

    return X_full_list, t_full_list, X_common_list, Y_common_list, t_common_list