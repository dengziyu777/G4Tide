import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def Fv6_evaluate_forecast(forecast_data_list, observation_data_list, output_file, note, print_on = True):
    """
    评估预报数据与实测数据（仅取观测值存在的时刻）- 支持每个站点独立数据结构

    参数:
        forecast_data_list: 预报潮位数据列表，每个元素是一个站点的数据数组 [时间戳, 潮位值]
        observation_data_list: 观测潮位数据列表，每个元素是一个站点的数据数组 [时间戳, 潮位值]
        output_file: 评估指标输出文件路径
        note: 评估说明（用于写入文件）
        print_on：执行函数时，是否打印信息
    """

    # 打开输出文件并写入标题
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"\n{note}\n")
        f.write("Site\tMAE\tRMSE\tR²\tSampleCount\n")

    # 检查站点数量是否一致
    if len(forecast_data_list) != len(observation_data_list):
        print(f"站点数量不匹配! 预报站点数: {len(forecast_data_list)}, 实测站点数: {len(observation_data_list)}")

    # 对每个站点单独处理
    for site_idx in range(len(forecast_data_list)):
        # 获取当前站点的预报和实测数据
        site_forecast = forecast_data_list[site_idx]
        site_observation = observation_data_list[site_idx]

        # 检查数据是否为空
        if len(site_forecast) == 0 or len(site_observation) == 0:
            print(f"站点 {site_idx + 1}: 无可用数据，跳过评估")
            continue

        # 提取时间戳和潮位值
        time_fc = site_forecast[:, 0]
        tide_fc = site_forecast[:, 1]
        time_obs = site_observation[:, 0]
        tide_obs = site_observation[:, 1]

        # 寻找时间重叠区间
        min_time = max(time_fc.min(), time_obs.min())
        max_time = min(time_fc.max(), time_obs.max())

        # 没有重叠时间段
        if min_time > max_time:
            print(
                f"站点 {site_idx + 1}: 预报({time_fc.min()}到{time_fc.max()})与观测({time_obs.min()}到{time_obs.max()})无重叠时间段")
            continue

        # 提取重叠时间段内的观测数据
        mask_obs = (time_obs >= min_time) & (time_obs <= max_time)
        time_obs_overlap = time_obs[mask_obs]
        tide_obs_overlap = tide_obs[mask_obs]

        # 对预报数据进行线性插值，使其对齐到观测数据的时间点
        sorted_idx_fc = np.argsort(time_fc)
        time_fc_sorted = time_fc[sorted_idx_fc]
        tide_fc_sorted = tide_fc[sorted_idx_fc]

        # 插值预报数据到观测时间点
        tide_fc_interp = np.interp(
            time_obs_overlap,
            time_fc_sorted,
            tide_fc_sorted,
            left=np.nan,
            right=np.nan
        )

        # 过滤掉插值失败的点（NaN值）
        valid_mask = ~np.isnan(tide_fc_interp)
        valid_time = time_obs_overlap[valid_mask]
        fc_interp = tide_fc_interp[valid_mask]
        obs_interp = tide_obs_overlap[valid_mask]

        # 没有有效点
        if len(valid_time) == 0:
            print(f"站点 {site_idx + 1}: 无法对齐任何数据点")
            continue

        # 计算评估指标
        mae = mean_absolute_error(obs_interp, fc_interp)
        rmse = np.sqrt(mean_squared_error(obs_interp, fc_interp))
        r2 = r2_score(obs_interp, fc_interp)

        # 写入评估结果
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"{site_idx + 1}\t{mae:.4f}\t{rmse:.4f}\t{r2:.4f}\t{len(valid_time)}\n")
        if print_on:
            print(f"站点 {site_idx + 1}: {len(valid_time)}个数据点评估完成")