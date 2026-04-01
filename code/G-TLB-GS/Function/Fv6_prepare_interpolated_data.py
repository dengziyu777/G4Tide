from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta
import numpy as np
import sys

def Fv6_prepare_interpolated_data(tide_data_list, meteo_data_list, site_params, print_on = True):
    """
    准备插值后的时序数据供主程序使用（支持历史/未来时间步长独立设置）

    参数:
        tide_data_list: 输入层预报数据列表
        meteo_data_list: 输入层环境数据列表
        site_params: 站点参数字典，格式为:
            {
                site_idx: {
                    "use_start_time": 起始时间 (YYYYMMDDHHMMSS),
                    "use_end_time": 结束时间 (YYYYMMDDHHMMSS),
                    "use_time_interval": 时间间隔 (秒),
                    "use_forward_hours": 历史时间步长 (小时),
                    "use_backward_hours": 未来时间步长 (小时)
                },
                ...
            }
        print_on：执行函数时，是否打印信息

    返回:
        tuple: (sequence_data_per_site, timestamps_per_site)
        sequence_data_per_site: 列表，每个站点的时间序列特征数据 [时序数, 特征数+1]
        timestamps_per_site: 列表，每个站点的Unix时间戳 [时序数]
    """

    # 0. 验证输入数据
    num_tide_sites = len(tide_data_list)
    num_meteo_sites = len(meteo_data_list)

    if num_tide_sites != num_meteo_sites:
        raise ValueError(f"潮位站点数({num_tide_sites})与气象站点数({num_meteo_sites})不一致")
    if print_on:
        print(f"站点数量: {num_tide_sites}个")

    # 用于存储结果的列表
    sequence_data_per_site = []
    timestamps_per_site = []

    # 1. 对每个站点独立处理
    for site_idx in range(num_tide_sites):
        if print_on:
            print(f"{'=' * 50}")
            print(f"开始处理站点 {site_idx + 1}/{num_tide_sites}")

        # 获取当前站点的参数
        if site_idx in site_params:
            site_param = site_params[site_idx]
            use_start_time = site_param["use_start_time"]
            use_end_time = site_param["use_end_time"]
            use_time_interval = site_param["use_time_interval"]
            history_steps = site_param["use_forward_hours"]
            future_steps = site_param["use_backward_hours"]
        else:
            raise KeyError(f"站点 {site_idx + 1} 没有在site_params中设置参数")

        # 1.1 解析起始和结束时间
        try:
            # 将时间数字转换为字符串
            start_str = str(use_start_time)
            end_str = str(use_end_time)

            time_formats = {
                14: "%Y%m%d%H%M%S", 12: "%Y%m%d%H%M",
                10: "%Y%m%d%H", 8: "%Y%m%d", 6: "%y%m%d"
            }

            start_len = len(start_str)
            start_fmt = time_formats.get(start_len, "%Y%m%d%H%M%S")
            end_len = len(end_str)
            end_fmt = time_formats.get(end_len, "%Y%m%d%H%M%S")

            # 解析时间字符串
            predict_start_dt = datetime.strptime(start_str, start_fmt)
            predict_end_dt = datetime.strptime(end_str, end_fmt)

            # 计算考虑历史步长的实际起始时间
            actual_start_dt = predict_start_dt - timedelta(hours=history_steps)
            actual_end_dt = predict_end_dt  # 因为未来时间步长是预测的，所以无需输入层中相应时段存在数据
            # actual_end_dt = predict_end_dt + timedelta(hours=future_steps)

        except Exception as e:
            print(f"错误: 解析站点 {site_idx + 1} 时间失败 - {e}")
            sys.exit(1)
            # predict_start_dt = datetime(1999, 5, 17)
            # predict_end_dt = datetime(2000, 5, 17)
            # actual_start_dt = predict_start_dt - timedelta(hours=0)
            # actual_end_dt = predict_end_dt + timedelta(hours=1)
            # print(f"警告: 使用默认时间范围")

        # 打印时间范围信息
        if print_on:
            print(f"站点 {site_idx + 1} 参数设置:")
            print(f"  DL模型设置起始结束时间: {predict_start_dt.strftime('%Y-%m-%d %H:%M:%S')} 至 {predict_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  DL模型历史步长: {history_steps}小时 | DL模型未来步长: {future_steps}小时")
            print(f"  DL模型实际数据利用时间: {actual_start_dt.strftime('%Y-%m-%d %H:%M:%S')} 至 {actual_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # 2. 检查当前站点数据时间范围
        tide_data = tide_data_list[site_idx]
        meteo_data = meteo_data_list[site_idx]

        # 数据范围检查函数
        def check_coverage(data, data_type, required_min, required_max):
            if data.size == 0:
                print(f"  警告: {data_type}数据为空")
                return required_min, required_max

            data_min = min(data[:, 0])
            data_max = max(data[:, 0])

            # 转换为可读时间
            min_dt = datetime.fromtimestamp(data_min).strftime("%Y-%m-%d %H:%M:%S")
            max_dt = datetime.fromtimestamp(data_max).strftime("%Y-%m-%d %H:%M:%S")
            req_min_dt = datetime.fromtimestamp(required_min).strftime("%Y-%m-%d %H:%M:%S")
            req_max_dt = datetime.fromtimestamp(required_max).strftime("%Y-%m-%d %H:%M:%S")

            # 检查是否覆盖要求的时间范围
            error_occurred = False

            if data_min > required_min:
                print(f"  错误: {data_type}数据开始时间({min_dt})晚于要求时间({req_min_dt})")
                error_occurred = True
            if data_max < required_max:
                print(f"  错误: {data_type}数据结束时间({max_dt})早于要求时间({req_max_dt})")
                error_occurred = True
            if print_on:
                print(f"  {data_type}数据时间范围: {min_dt} - {max_dt}")

            # 如果有错误，立即退出
            if error_occurred:
                sys.exit(1)

            return data_min, data_max

        # 检查输入层预报数据范围
        tide_min, tide_max = check_coverage(tide_data, "输入层预报",actual_start_dt.timestamp(),
                                            actual_end_dt.timestamp())

        # 检查输入层环境数据范围
        meteo_min, meteo_max = check_coverage(meteo_data, "输入层环境场",actual_start_dt.timestamp(),actual_end_dt.timestamp())

        # 3. 创建当前站点的目标时间网格
        # 计算总秒数
        total_seconds = int((actual_end_dt - actual_start_dt).total_seconds())

        # 计算时间步数
        num_steps = total_seconds // use_time_interval      # 计算时间步数

        # 创建时间戳数组
        target_timestamps = np.array([
            (actual_start_dt + timedelta(seconds=i * use_time_interval)).timestamp()    # 计算每个时间点的具体时间
            for i in range(num_steps + 1)   # 创建从0到num_steps（包含）的整数序列
        ])

        # print(f"  创建时间网格: {len(target_timestamps)}个时间步 ({num_steps}步)")

        # 4. 验证特征数量
        num_meteo_features = meteo_data.shape[1] - 1 if meteo_data.size > 0 else 0
        total_features = 1 + num_meteo_features  # 潮位 + 气象特征

        # 5. 插值处理
        # 5.1 插值输入层预报数据
        if tide_data.size > 0:
            cs_tide = CubicSpline(tide_data[:, 0], tide_data[:, 1])
            interp_tide = cs_tide(target_timestamps)
        else:
            print("  警告: 输入层预报数据为空，使用默认值0")
            interp_tide = np.zeros(len(target_timestamps))

        # 5.2 插值输入层环境场数据
        interp_meteo = np.zeros((len(target_timestamps), num_meteo_features))
        for feat_idx in range(num_meteo_features):
            if meteo_data.size > 0:
                cs_meteo = CubicSpline(meteo_data[:, 0], meteo_data[:, 1 + feat_idx])
                interp_meteo[:, feat_idx] = cs_meteo(target_timestamps)
            else:
                print(f"  警告: 输入层环境场特征 {feat_idx} 数据为空，使用默认值0")

        # 5.3 组合数据
        # 创建完整的时序数据
        full_sequence = np.zeros((len(target_timestamps), total_features))
        full_sequence[:, 0] = interp_tide  # 潮位作为第一特征
        if num_meteo_features > 0:
            full_sequence[:, 1:1 + num_meteo_features] = interp_meteo

        # 直接使用完整序列作为核心序列
        core_sequence = full_sequence
        core_timestamps = target_timestamps

        # 记录时间范围信息
        predict_start_dt = datetime.strptime(str(use_start_time), start_fmt)
        predict_end_dt = datetime.strptime(str(use_end_time), end_fmt)

        # print(f"  实际处理时间范围: ")
        # print(f"    起始: {datetime.fromtimestamp(target_timestamps[0]).strftime('%Y-%m-%d %H:%M')}")
        # print(f"    结束: {datetime.fromtimestamp(target_timestamps[-1]).strftime('%Y-%m-%d %H:%M')}")
        # print(f"    包含历史数据: {history_steps}小时")
        # print(f"    包含未来预测: {future_steps}小时")

        # 保存当前站点结果
        sequence_data_per_site.append(core_sequence)
        timestamps_per_site.append(core_timestamps)

        if print_on:
        # print(f"  完成处理: 总时间步数={len(core_timestamps)}, 时间步长={use_time_interval}s, 总特征数={total_features}")
            print(f"{'=' * 50}")
    if print_on:
        print("所有站点处理完成")
    return sequence_data_per_site, timestamps_per_site