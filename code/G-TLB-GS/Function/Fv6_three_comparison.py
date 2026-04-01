import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from source.Fv6_generate_time_ticks import Fv6_generate_time_ticks


def Fv6_three_comparison(site_batch1_data, site_predictions, site_predict_timestamps,
                         site_batch2_data, site_params, output_prediction_plot_folder, title, ylabel,
                         time_interval_hours, rotation_user, output_evaluation_metrics_path,plot_common_period_only,
                         batch3_data_folder_path):
    """
    绘制输入层预报数据、输出层预报数据和实测数据的对比图（适应新数据格式）
    分上下两个子图分别比较：
    上：Observed vs Forecast
    下：Observed vs G-PREC-C2C

    参数:
        site_batch1_data: 列表，每个元素是一个站点的输入层预报数据 [[时间戳, 预报值], ...]
        site_predictions: 字典，每个站点的模型预测结果 {站点索引: 预测值数组}
        site_predict_timestamps: 字典，每个站点的预测时间戳 {站点索引: 时间戳数组}
        site_batch2_data: 列表，每个元素是一个站点的实测数据 [[时间戳, 实测值], ...]
        site_params: 字典，每个站点的参数设置 {站点索引: 参数字典}
        output_prediction_plot_folder: 对比图输出文件夹
        title: 图表标题
        ylabel: Y轴标签
        time_interval_hours: X轴时间间隔（小时）
        rotation_user: X轴标签旋转角度
        output_evaluation_metrics_path: 评估结果文件输出路径
        plot_common_period_only: 是否仅绘制公共时间段的数据，默认为True
        batch3_data_folder_path：ERA5文件夹，用于SHAP分析结果命名
    """
    # 0.0获取文件夹中的所有文件名（去除后缀），并按升序排序
    file_names = []
    if os.path.isdir(batch3_data_folder_path):
        for filename in os.listdir(batch3_data_folder_path):
            if os.path.isfile(os.path.join(batch3_data_folder_path, filename)):
                # 去除文件后缀
                base_name = os.path.splitext(filename)[0]
                file_names.append(base_name)

    # 按文件名升序排序
    file_names.sort()

    # 0.1 全局设置 Times New Roman 字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 1. 获取站点数量
    num_sites = len(site_batch1_data)

    # 2. 准备评估指标
    metrics_header = "站点ID 输入层预报_MAE RMSE R²\n" \
                     "站点ID 输出层预报_MAE RMSE R²\n"
    metrics_content = ""

    # 4. 遍历每个站点
    plt.ioff()
    all_metrics = []  # 准备评估指标存储列表（新增）
    for site_idx in range(num_sites):
        try:
            # 5. 创建带两个子图的图形（上下排列）
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            fig.subplots_adjust(hspace=0.1)  # 减少子图间距

            # 获取当前站点的文件名（按升序排序）
            if site_idx < len(file_names):
                file_title = file_names[site_idx]
            else:
                # 如果站点数多于文件名数，循环使用文件名
                file_title = file_names[site_idx % len(file_names)]

            # 获取当前站点的数据
            batch1_data = site_batch1_data[site_idx]  # [[时间戳, 预报值], ...]
            predictions = site_predictions[site_idx]  # 预测值数组
            predict_ts = site_predict_timestamps[site_idx]  # 预测时间戳数组
            batch2_data = site_batch2_data[site_idx] if site_batch2_data else None  # [[时间戳, 实测值], ...]
            site_param = site_params[site_idx]  # 获取当前站点的参数
            history_steps = site_param["use_forward_hours"]

            # 设置主标题
            fig.suptitle(f"-{history_steps}h / {title}{file_title}", fontsize=18)

            # 6. 提取公共时间段数据（如果启用）
            if plot_common_period_only:
                # 获取各数据集的时间范围
                b1_min_ts = batch1_data[:, 0].min()
                b1_max_ts = batch1_data[:, 0].max()
                pred_min_ts = predict_ts.min()
                pred_max_ts = predict_ts.max()

                # 计算公共时间段
                common_min_ts = max(b1_min_ts, pred_min_ts)
                common_max_ts = min(b1_max_ts, pred_max_ts)

                if batch2_data is not None:
                    b2_min_ts = batch2_data[:, 0].min()
                    b2_max_ts = batch2_data[:, 0].max()
                    common_min_ts = max(common_min_ts, b2_min_ts)
                    common_max_ts = min(common_max_ts, b2_max_ts)

                # 验证公共时间段有效性
                if common_min_ts > common_max_ts:
                    print(f"  警告: 站点 {file_title} 无重叠时间段，使用完整数据")
                    common_min_ts = min(b1_min_ts, pred_min_ts)
                    common_max_ts = max(b1_max_ts, pred_max_ts)

                # 提取公共时间段数据
                def extract_common_data(data):
                    mask = (data[:, 0] >= common_min_ts) & (data[:, 0] <= common_max_ts)
                    return data[mask]

                batch1_data = extract_common_data(batch1_data)
                predict_mask = (predict_ts >= common_min_ts) & (predict_ts <= common_max_ts)
                predict_ts = predict_ts[predict_mask]  # 公共时间戳
                predictions = predictions[predict_mask]

                if batch2_data is not None:
                    batch2_data = extract_common_data(batch2_data)

                # print(f"  站点 {file_title} 公共时间段: "
                #       f"{datetime.fromtimestamp(common_min_ts).strftime('%Y-%m-%d %H:%M')} "
                #       f"至 {datetime.fromtimestamp(common_max_ts).strftime('%Y-%m-%d %H:%M')}")

            # 7. 在上方子图(ax1)绘制：Observed vs Forecast
            # 7.1 绘制实测数据（如果存在）
            if batch2_data is not None:
                b2_timestamps = batch2_data[:, 0]
                b2_values = batch2_data[:, 1]
                ax1.plot(b2_timestamps, b2_values,
                         label='Observed', color='#1f77b4', alpha=1, linewidth=1.5)   # 蓝色

            # 7.2 绘制完整的输入层预报数据（包括历史+公共时段）
            # 绘制Forecast
            b1_timestamps = batch1_data[:, 0]
            b1_values = batch1_data[:, 1]
            ax1.plot(batch1_data[:,0], batch1_data[:,1],
                     label='Baseline Plan', color='#2ca02c', alpha=1, linewidth=1.5)    # 绿色
            ax1.set_ylabel(ylabel)
            ax1.grid(True, linestyle=':', alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.set_title('Observed vs Baseline Plan', fontsize=18)
            ax1.set_ylabel(ylabel, fontsize=16)  # 设置y轴标签字号
            ax1.legend(loc='upper left', fontsize=16)  # 设置图例字号
            ax1.tick_params(axis='both', labelsize=14)  # 设置坐标轴刻度字号

            # 8. 在下方子图(ax2)绘制：Observed vs G-TLB-GS
            # 8.1 绘制实测数据（如果存在）
            if batch2_data is not None:
                ax2.plot(b2_timestamps, b2_values, label='Observed', color='#1f77b4', alpha=1, linewidth=1.5)   # 蓝色

            # 8.2 绘制输出层预报数据
            # 绘制G-TLB-GS
            ax2.plot(predict_ts, predictions,
                     label='G-TLB-GS Forecast', color='#ff7f0e', alpha=1, linewidth=1.5)  # 橙色
            ax2.set_ylabel(ylabel)
            ax2.grid(True, linestyle=':', alpha=0.3)
            ax2.legend(loc='upper left')
            ax2.set_title('Observed vs G-TLB-GS Forecast', fontsize=18)
            ax2.set_ylabel(ylabel, fontsize=16)  # 设置y轴标签字号
            ax2.legend(loc='upper left', fontsize=16)  # 设置图例字号
            ax2.tick_params(axis='both', labelsize=14)  # 设置坐标轴刻度字号

            # 9. 设置共享的X轴
            min_ts = min(batch1_data[:, 0].min(), predict_ts.min())
            max_ts = max(batch1_data[:, 0].max(), predict_ts.max())
            if batch2_data is not None:
                min_ts = min(min_ts, b2_timestamps.min())
                max_ts = max(max_ts, b2_timestamps.max())

            tick_locations, tick_labels = Fv6_generate_time_ticks(min_ts, max_ts, time_interval_hours)
            ax2.set_xticks(tick_locations)
            ax2.set_xticklabels(tick_labels, rotation=rotation_user)
            ax2.set_xlim(min_ts, max_ts)
            fig.text(0.5, 0.01, 'Time', ha='center', fontsize=16)  # 共享的X轴标签

            # 10. 保存图像
            plt.tight_layout(rect=[0, 0.01, 1, 0.98])  # 为标题留空间；用于指定一个矩形区域，表示在调整子图布局时，所有子图应该被约束在这个矩形之内
            output_file = os.path.join(output_prediction_plot_folder, f"{file_title}_comparison.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # 11. 计算评估指标（如果有实测数据）
            if batch2_data is not None:
                # 时间对齐处理
                time_base = b2_timestamps

                # 输入层预报数据插值
                interp_b1 = np.interp(time_base, b1_timestamps, b1_values)

                # 输出层预报数据插值
                interp_output = np.interp(time_base, predict_ts, predictions)

                # 计算指标
                def calculate_metrics(y_true, y_pred):
                    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]

                    if len(y_true) == 0:
                        return np.nan, np.nan, np.nan

                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    return mae, rmse, r2

                # 计算输入层预报指标
                mae_b1, rmse_b1, r2_b1 = calculate_metrics(b2_values, interp_b1)

                # 计算输出层预报指标
                mae_out, rmse_out, r2_out = calculate_metrics(b2_values, interp_output)

                # 添加到指标内容（修改为存储字典格式）
                site_metrics = {
                    "site_id": file_title,
                    "b1_mae": mae_b1,
                    "b1_rmse": rmse_b1,
                    "b1_r2": r2_b1,
                    "out_mae": mae_out,
                    "out_rmse": rmse_out,
                    "out_r2": r2_out
                }
                all_metrics.append(site_metrics)

                # 添加到指标内容
                metrics_content += (
                    f"{file_title} {mae_b1:.4f} {rmse_b1:.4f} {r2_b1:.4f}\n"
                    f"{file_title} {mae_out:.4f} {rmse_out:.4f} {r2_out:.4f}\n"
                )

            else:
                print(f"  站点 {file_title} 无实测数据，无法计算指标")

        except Exception as e:
            print(f"站点 {file_title} 对比图生成失败: {str(e)}")

    # 12. 保存评估指标（增加平均值计算）
    if metrics_content:
        # 计算所有站点平均值（新增）
        avg_b1_mae = np.mean([m['b1_mae'] for m in all_metrics])
        avg_b1_rmse = np.mean([m['b1_rmse'] for m in all_metrics])
        avg_b1_r2 = np.mean([m['b1_r2'] for m in all_metrics])
        avg_out_mae = np.mean([m['out_mae'] for m in all_metrics])
        avg_out_rmse = np.mean([m['out_rmse'] for m in all_metrics])
        avg_out_r2 = np.mean([m['out_r2'] for m in all_metrics])

        # 添加平均值到输出内容（新增）
        metrics_content += "\n# Site Averages\n"
        metrics_content += f"Average_INPUT {avg_b1_mae:.4f} {avg_b1_rmse:.4f} {avg_b1_r2:.4f}\n"
        metrics_content += f"Average_OUTPUT {avg_out_mae:.4f} {avg_out_rmse:.4f} {avg_out_r2:.4f}\n"

        with open(output_evaluation_metrics_path, 'w', encoding='utf-8') as f:
            f.write(metrics_header)
            f.write(metrics_content)
        print(f"评估指标已保存至: {output_evaluation_metrics_path}")
    else:
        print("警告: 无实测数据，未生成评估指标")