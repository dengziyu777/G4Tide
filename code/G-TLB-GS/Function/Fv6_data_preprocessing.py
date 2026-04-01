import numpy as np
from datetime import datetime
from source.Fv6_adjust_observation_interval import Fv6_adjust_observation_interval


def Fv6_adjust_and_prepare_observation(obs_data_list, new_interval, print_on = True):
    """
    调整观测数据时间间隔并准备数据 - 支持每个站点独立数据结构

    参数:
        obs_data_list: 观测数据列表，每个元素是一个站点的数据数组 [时间戳, 潮位值]
        new_interval: 新时间间隔 (秒)
        print_on：执行函数时，是否打印信息

    返回:
        list: 调整后的观测数据列表，每个元素是一个站点的数据数组
    """
    adjusted_obs_list = []

    for site_idx, site_data in enumerate(obs_data_list):
        # 检查数据是否为空
        if len(site_data) == 0:
            print(f"站点 {site_idx + 1}: 无观测数据，跳过调整")
            adjusted_obs_list.append(np.empty((0, 2)))
            continue

        # 提取时间戳和潮位值
        time = site_data[:, 0]
        tide = site_data[:, 1]  # 单列潮位值

        # 1. 从站点数据推导原始时间间隔
        # 计算所有连续时间戳差值的中位数
        time_diffs = np.diff(site_data[:, 0])

        if len(time_diffs) > 0:
            # 取中位数并四舍五入为整数
            orig_interval = int(np.round(np.median(time_diffs)))
            if print_on:
                print(f"站点 {site_idx+1}: 自动推导观测原始时间间隔 = {orig_interval}秒；新时间间隔 = {new_interval}秒")
        else:
            # 当只有一个数据点时，无法推导时间间隔
            print(f"站点 {site_idx+1}: 只有一个数据点，无法计算时间间隔，跳过调整")
            adjusted_obs_list.append(site_data.copy())
            continue

        # 执行插值调整
        time_adjusted, tide_adjusted = Fv6_adjust_observation_interval(
            time, tide, orig_interval, new_interval
        )

        # 创建调整后的数据数组
        site_adjusted = np.column_stack((time_adjusted, tide_adjusted))
        adjusted_obs_list.append(site_adjusted)

        if print_on:
            print(f"站点 {site_idx + 1}: 原始时序数 {len(time)} → 调整后时序数 {len(time_adjusted)}")

    return adjusted_obs_list


def Fv6_extract_overlap_period(forecast_data_list, obs_data_adjusted_list, forward_hours, print_on = True):
    """
    提取预报和观测数据的重叠时间段，考虑预测数据提前的情况 - 支持每个站点独立数据结构

    参数:
        forecast_data_list: 预报数据列表，每个元素是一个站点的数据数组 [时间戳, 潮位值]
        obs_data_adjusted_list: 调整后的观测数据列表，每个元素是一个站点的数据数组 [时间戳, 潮位值]
        forward_hours: 预测数据较观测数据的提前小时数（整数）
        print_on：执行函数时，是否打印信息

    返回:
        tuple: (
            forecast_overlap_list,  # 重叠时间段的原始预报数据列表
            obs_overlap_list        # 重叠时间段的观测数据列表
        )
    """
    forecast_overlap_list = []
    obs_overlap_list = []

    # 确保forward_hours是整数
    forward_hours = int(forward_hours)
    print(f"预测提前量: {forward_hours}小时")

    # 检查站点数量是否一致
    if len(forecast_data_list) != len(obs_data_adjusted_list):
        print(f"站点数量不匹配! 预报站点数: {len(forecast_data_list)}, 观测站点数: {len(obs_data_adjusted_list)}")
        return forecast_overlap_list, obs_overlap_list

    for site_idx in range(len(forecast_data_list)):
        # 获取当前站点的预报和观测数据
        fc_data = forecast_data_list[site_idx]
        obs_data = obs_data_adjusted_list[site_idx]

        # 检查数据是否为空
        if len(fc_data) == 0 or len(obs_data) == 0:
            print(f"站点 {site_idx + 1}: 无可用数据，跳过")
            forecast_overlap_list.append(np.empty((0, 2)))
            obs_overlap_list.append(np.empty((0, 2)))
            continue

        # 提取时间戳和潮位值
        time_fc = fc_data[:, 0]     # 原始预报数据时间范围
        tide_fc = fc_data[:, 1]
        time_obs = obs_data[:, 0]   # 观测数据时间范围
        tide_obs = obs_data[:, 1]

        # 推导时间间隔（整数秒）
        fc_interval = int(np.round(np.median(np.diff(time_fc))))
        obs_interval = int(np.round(np.median(np.diff(time_obs))))

        if print_on:
            print(f"站点 {site_idx + 1}: 原始预报间隔 {fc_interval}s, 插值观测间隔 {obs_interval}s")

        # 寻找时间重叠区间
        min_time_orig = max(time_fc.min(), time_obs.min())
        max_time = min(time_fc.max(), time_obs.max())
        min_time = min_time_orig - forward_hours * 3600  # 转换为秒

        # 没有重叠时间段
        if min_time > max_time:
            time_min_fc = datetime.fromtimestamp(time_fc.min())
            time_max_fc = datetime.fromtimestamp(time_fc.max())
            time_min_obs = datetime.fromtimestamp(time_obs.min())
            time_max_obs = datetime.fromtimestamp(time_obs.max())
            if print_on:
                print(f"站点 {site_idx + 1}: 预报({time_min_fc.strftime('%Y-%m-%d %H:%M')}到"
                    f"{time_max_fc.strftime('%Y-%m-%d %H:%M')})与观测({time_min_obs.strftime('%Y-%m-%d %H:%M')}到"
                    f"{time_max_obs.strftime('%Y-%m-%d %H:%M')})无重叠时间段")
            forecast_overlap_list.append(np.empty((0, 2)))
            obs_overlap_list.append(np.empty((0, 2)))
            continue

        # 提取重叠时间段数据
        mask_fc = (time_fc >= min_time) & (time_fc <= max_time)
        mask_obs = (time_obs >= min_time_orig) & (time_obs <= max_time)

        time_fc_overlap = time_fc[mask_fc]
        tide_fc_overlap = tide_fc[mask_fc]
        time_obs_overlap = time_obs[mask_obs]
        tide_obs_overlap = tide_obs[mask_obs]

        # 创建重叠时间段的数据数组
        fc_overlap = np.column_stack((time_fc_overlap, tide_fc_overlap))
        obs_overlap = np.column_stack((time_obs_overlap, tide_obs_overlap))

        forecast_overlap_list.append(fc_overlap)    # 原始预报数据公共时段内数据（包含提前考虑时间）
        obs_overlap_list.append(obs_overlap)        # 观测数据公共时段内数据

        # 使用可读时间格式显示时间范围
        start_str = datetime.fromtimestamp(min_time_orig).strftime('%Y-%m-%d %H:%M')
        end_str = datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M')

        if print_on:
            print(f"站点 {site_idx + 1}: 重叠时间范围 {start_str} 到 {end_str} // "
                   f"重叠时间内预报点: {len(time_fc_overlap)}，观测点: {len(time_obs_overlap)}")

    return forecast_overlap_list, obs_overlap_list