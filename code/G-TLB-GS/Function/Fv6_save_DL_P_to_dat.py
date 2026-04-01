import os
import numpy as np
import pandas as pd
from datetime import datetime


def Fv6_save_DL_P_to_dat(site_predictions, site_predict_timestamps, output_dir, case_name):
    """
    将每个站点的预测数据保存到单独的.dat文件和一个整合所有站点的文件
    修复问题：
    1. 统一站点索引系统（显示时+1但存储用原始索引）
    2. 解决键值类型不匹配问题
    3. 优化时间戳合并逻辑

    参数:
        site_predictions: 字典，每个站点的预测结果 {站点索引: 预测值数组}
        site_predict_timestamps: 字典，每个站点的预测时间戳 {站点索引: 时间戳数组}
        output_dir: 输出目录
        case_name：案例名称
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 过滤掉空数据的站点
    valid_sites = []
    for site_idx in site_predictions:
        if (len(site_predictions[site_idx]) > 0 and
                len(site_predict_timestamps[site_idx]) > 0 and
                len(site_predictions[site_idx]) == len(site_predict_timestamps[site_idx])):
            valid_sites.append(site_idx)
        else:
            print(f"警告: 站点 {site_idx} 数据为空或长度不一致，将被跳过")

    if not valid_sites:
        print("错误: 没有有效的站点数据可供保存")
        return

    # 按索引排序有效站点
    site_indices = sorted(valid_sites)

    # 准备所有站点数据整合
    all_timestamps_set = set()
    all_sites_data = {}

    # 第一阶段：收集所有时间戳和处理单文件
    for site_idx in site_indices:
        try:
            # 获取当前站点数据
            predictions = site_predictions[site_idx]
            predict_ts = site_predict_timestamps[site_idx]

            # 保存单站点文件
            site_filename = f"{case_name}_site{site_idx + 1}_DL_P.dat"
            site_filepath = os.path.join(output_dir, site_filename)

            with open(site_filepath, 'w', encoding='utf-8') as f:
                f.write("DateTime\tDL_Forecast\n")
                for ts, value in zip(predict_ts, predictions):
                    dt_str = datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
                    f.write(f"{dt_str}\t{value:.6f}\n")
            # print(f"  站点 {site_idx + 1} 预测数据已保存至: {site_filepath}")

            # 收集站点数据和时间戳
            all_sites_data[site_idx] = {
                "timestamps": predict_ts,
                "values": predictions
            }

            # 添加到全局时间戳集合
            all_timestamps_set.update(predict_ts)

        except Exception as e:
            print(f"站点 {site_idx + 1} 数据保存失败: {str(e)}")

    # 第二阶段：创建合并文件
    if all_timestamps_set:
        try:
            # 创建按时间排序的时间戳列表
            sorted_timestamps = sorted(all_timestamps_set)

            # 准备DataFrame数据
            data_dict = {"DateTime": [datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
                                      for ts in sorted_timestamps]}

            # 为每个站点添加列数据
            for site_idx in site_indices:
                site_data = all_sites_data[site_idx]

                # 创建快速查找字典（时间戳->预测值）
                value_map = dict(zip(site_data["timestamps"], site_data["values"]))

                # 添加列数据，缺失数据为NaN
                data_dict[f"Site_{site_idx + 1}"] = [value_map.get(ts, np.nan)
                                                     for ts in sorted_timestamps]

            # 创建并保存DataFrame
            df = pd.DataFrame(data_dict)
            all_sites_filename = f"{case_name}_all_sites_DL_P.dat"
            all_sites_filepath = os.path.join(output_dir, all_sites_filename)

            df.to_csv(all_sites_filepath, sep='\t', index=False, float_format='%.6f', na_rep='NaN')
            print(f"  合并数据已保存至: {all_sites_filepath}")
            print(f"  包含 {len(site_indices)} 个站点, {len(sorted_timestamps)} 个时间点")

        except Exception as e:
            print(f"合并数据保存失败: {str(e)}")
    else:
        print("警告: 未生成合并文件，因为未找到有效时间戳")

    print(f"数据处理完成，输出目录: {output_dir}")