import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from joblib import load

from source.Fv6_safe_torch_load import Fv6_safe_torch_load
from source.Fv6_TCNModel import TCNModel
from source.Fv6_LSTMModel import LSTMModel


def Fv6_SHAP_analysis_per_site(
        model_type, model_save_path, case_name,
        train_loader, test_loader, input_feature_names, site_idx,
        feature_scaler, shap_n_background, shap_max_samples, output_dir,
        input_steps_to_analyze, output_steps_to_analyze,
        forward_hours, batch2_data_time_interval_adjust, forward_steps_4SHAP,
        backwards_steps, backwards_steps_4SHAP,batch3_data_folder_path
):
    """
    单站点SHAP特征贡献度分析实现函数（v6版本）
    250715：修改命名规则，按照batch3_data_folder_path中子文件命名

    参数说明：
        model_type: 模型类型（TCN或LSTM）
        model_save_path: 模型保存目录
        case_name: 案例名称
        train_loader: 当前站点的训练数据加载器
        test_loader: 当前站点的测试数据加载器
        input_feature_names: 输入特征名称列表
        site_idx: 站点索引
        feature_scaler: 当前站点的特征标准化器
        shap_n_background: SHAP背景样本数量
        shap_max_samples: SHAP分析样本数量
        output_dir: 分析结果输出目录
        input_steps_to_analyze: 要分析的输入时间步索引（列表或"auto"）
        output_steps_to_analyze: 要分析的输出时间步索引（列表或"auto"）
        forward_hours: 历史时间步长（小时）
        batch2_data_time_interval_adjust: 时间步长（秒）
        forward_steps_4SHAP: SHAP分析时选择的代表性历史时间步数量
        backwards_steps: 未来时间步长（小时）
        backwards_steps_4SHAP: SHAP分析时选择的代表性未来时间步数量
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

    # 获取当前站点的文件名（按升序排序）
    if site_idx < len(file_names):
        file_title = file_names[site_idx]
    else:
        # 如果站点数多于文件名数，循环使用文件名
        file_title = file_names[site_idx % len(file_names)]

    # 创建子文件夹用于存储
    output_subfolder = os.path.join(output_dir, file_title)

    # 0.1设置评估设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0.2统一设置所有图表的分辨率和格式参数
    plt.rcParams.update({
        'savefig.dpi': 300,  # 全局设置保存分辨率（300 DPI）
        'savefig.format': 'png',  # 全局设置保存格式（PNG）
        'savefig.bbox': 'tight',  # 全局设置自动裁剪（相当于bbox_inches='tight'）
        'figure.figsize': (15, 8),  # 全局设置图像大小（宽度15英寸，高度8英寸）
    })

    # 模型文件路径
    model_filename = f"{model_type}_{case_name}.pth"
    model_fullpath = os.path.join(model_save_path, model_filename)
    model_config_name = f"{model_type}_{case_name}_config.pkl"
    model_config_path = os.path.join(model_save_path, model_config_name)

    # 1. 准备数据加载器中的样本
    def extract_samples(loader, max_samples):  # 用于从数据加载器 (loader) 中提取指定数量的样本 (max_samples)
        X_samples = []
        for inputs, _ in loader:  # 遍历数据加载器中的每个批次
            X_samples.append(inputs)
            if len(torch.cat(X_samples)) >= max_samples:
                break
        return torch.cat(X_samples)[:max_samples]  # 将列表中所有批次的张量连接起来，并截取前max_samples个样本

    # 2. 加载模型配置并重建模型
    try:
        model_config = load(model_config_path)
        print(f"--SHAP，成功加载模型配置: {model_config_path}")

        if model_type == 'TCN':  # 判断是否为TCN模型
            model = TCNModel(
                input_size=model_config['input_size'],
                output_size=model_config['output_size'],
                num_channels=model_config['num_channels'],
                kernel_size=model_config['kernel_size'],
                dropout=model_config['dropout']
            )
        elif model_type == 'LSTM':  # 判断是否为LSTM模型
            model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                output_size=model_config['output_size'],
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout']
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        # 安全加载模型权重
        model.load_state_dict(Fv6_safe_torch_load(model_fullpath, device))
        model.to(device)
        model.eval()
        print(f"--SHAP，成功加载模型: {model_fullpath}")

    except Exception as e:
        print(f"--SHAP，加载模型失败: {e}")
        return

    # 3. 准备背景数据和解释数据
    try:
        X_background = extract_samples(train_loader, shap_n_background)  # 背景数据，得基线值；样本数*数据长度*特征总数
        X_explain = extract_samples(test_loader, shap_max_samples)  # 解释数据，得最终预测值；样本数*数据长度*特征总数

        X_background = X_background.to(device)
        X_explain = X_explain.to(device)

        # 获取数据维度
        if X_background.dim() == 3:
            n_bg_samples, seq_len, n_features = X_background.shape
        else:
            n_bg_samples, seq_len = X_background.shape[:2]
            n_features = 1

        print(f"  SHAP，背景数据形状: {X_background.shape}")
        print(f"  SHAP，解释数据形状: {X_explain.shape}")

    except Exception as e:
        print(f"  SHAP，数据处理失败: {str(e)}")
        return

    # 4. 处理'auto'参数逻辑
    # 输入步处理
    if input_steps_to_analyze == "auto":
        num_steps = seq_len
        num_to_select = min(forward_steps_4SHAP, seq_len)  # 最多选择forward_steps_4SHAP个代表性时间步

        if num_steps <= num_to_select:
            # 如果时间步数量少于或等于要选择的数量，使用所有时间步
            input_steps_to_analyze = list(range(num_steps))
        else:
            # 生成等间距的浮点索引
            float_indices = np.linspace(0, num_steps - 1, num_to_select)
            # 四舍五入为整数索引
            step_indices = np.round(float_indices).astype(int)
            # 去重并排序
            unique_sorted_indices = sorted(set(step_indices))

            # 如果去重后数量不足，补充缺失的索引
            if len(unique_sorted_indices) < num_to_select:
                missing_count = num_to_select - len(unique_sorted_indices)
                all_possible = set(range(num_steps))
                missing_indices = sorted(all_possible - set(unique_sorted_indices))[:missing_count]
                input_steps_to_analyze = sorted(unique_sorted_indices + missing_indices)
            else:
                input_steps_to_analyze = unique_sorted_indices

        print(f"  SHAP，自动等间隔选择{num_to_select}个输入层时间步: {input_steps_to_analyze}")  # 由历史的时间步到现在的

    # 输出步处理
    if output_steps_to_analyze == "auto":
        # 测试一个样本以获取输出序列长度
        with torch.no_grad():
            test_output = model(X_explain[:1])
            output_seq_len = test_output.shape[1]

        # 自动选择代表性输出时间步
        if output_seq_len <= backwards_steps_4SHAP:
            output_steps_to_analyze = list(range(output_seq_len))
        else:
            # 生成等间距索引
            output_steps_to_analyze = np.round(
                np.linspace(0, output_seq_len - 1, backwards_steps_4SHAP)
            ).astype(int).tolist()

        print(f"  SHAP，自动选择{len(output_steps_to_analyze)}个输出时间步: {output_steps_to_analyze}")

    # 确保步索引是列表
    if not isinstance(input_steps_to_analyze, list):
        input_steps_to_analyze = [input_steps_to_analyze]
    if not isinstance(output_steps_to_analyze, list):
        output_steps_to_analyze = [output_steps_to_analyze]

    # 5. 创建GradientExplainer
    try:
        explainer = shap.GradientExplainer(model, X_background)  # 基准值base_values隐含在梯度积分路径中
        print("  SHAP，成功创建SHAP解释器")
    except Exception as e:
        print(f"  SHAP，创建SHAP解释器失败: {e}")
        return

    # 6. 计算SHAP值 - 关键修改：在训练模式下计算
    model.train()  # 将模型暂时设为训练模式

    # 获取SHAP值
    shap_values_all = explainer(X_explain)
    print(f"  SHAP，shap_values_all形状: {shap_values_all.shape}")  # SHAP值会对"训练样本数量*输入层需考虑时序数量×特征数"计算

    # 针对每一个输出时间步进行解释===================
    for out_step in output_steps_to_analyze:

        backward_h = int(out_step * batch2_data_time_interval_adjust / 3600)  # 当前分析输出时间步代表未来h
        print(f"{'-' * 2} 分析输出步 {out_step}，即+{backward_h}h ")

        # 计算所有训练样本的预测值，并取平均
        with torch.no_grad():
            # 一次前向传播处理整个背景数据集
            all_pred = model(X_background)  # 形状：[n_samples, ...]
            print(f"  SHAP，all_pred形状: {all_pred.shape}")
            all_pred_out_step = all_pred[:,out_step]  # 取当前处理输出时间步 预测结果

            # 计算整个数据集上的均值，确保结果为 NumPy float32
            base_value = np.float32(all_pred_out_step.mean().item())
        print(f"  SHAP，base_value：{base_value:.3f}")

        # 将模型设回评估模式
        model.eval()

        # 根据模型类型处理SHAP值
        if len(output_steps_to_analyze) == 1:
            shap_out_steps_values = shap_values_all.values  # 所计算的SHAP值
            shap_all_data = shap_values_all.data    # 原始数据
        else:
            shap_out_steps_values = shap_values_all.values[out_step]  # 若shap_values_all形状(13, 5, 25, 11)，该操作后取第0维
            shap_all_data = shap_values_all.data
        print(f"  SHAP，shap_out_steps_values: {shap_out_steps_values.shape}")
        print(f"  SHAP，shap_all_data: {shap_all_data.shape}")

        # 7. 选择特定的输入时间步，同时保持Explanation对象类型
        try:

            # 获取原始SHAP值的数据
            original_values = shap_out_steps_values
            original_data = shap_all_data

            # 获取反标准化的数据值
            original_data_np = original_data.cpu().numpy()
            n_samples, n_steps, n_features = original_data_np.shape

            # 反标准化输入数据
            original_data_reshaped = original_data_np.reshape(-1, n_features)
            original_data_destandardized = feature_scaler.inverse_transform(original_data_reshaped)    # X反标准化
            original_data_destandardized = original_data_destandardized.reshape(n_samples, n_steps, n_features)

            # 选择特定时间步的SHAP值和数据
            selected_values = original_values[:, input_steps_to_analyze, :]
            selected_data = original_data_destandardized[:, input_steps_to_analyze, :]

            # 重构新的Explanation对象
            n_samples = selected_values.shape[0]
            shap_values_selected = shap.Explanation(
                values=selected_values,
                base_values=base_value,  # 使用计算得到的基准值
                data=selected_data,
                feature_names=[f"Feature_{i}" for i in range(selected_data.shape[-1])]  # 临时特征名，后续替换
            )
        except Exception as e:
            print(f"  SHAP，重构SHAP对象失败: {e}")
            return   # 直接退出整个Fv6_SHAP_analysis_per_site函数

        # 8. 展平特征维度 (保持Explanation对象)
        try:
            # 获取展平后的SHAP值
            n_steps = shap_values_selected.values.shape[1]
            n_features = shap_values_selected.values.shape[2]
            shap_values_flat = shap_values_selected.values.reshape(n_samples, n_steps * n_features)

            # 获取展平后的数据值
            data_flat = shap_values_selected.data.reshape(n_samples, n_steps * n_features)

            # 重构展平后的Explanation对象
            shap_values_2d = shap.Explanation(
                values=shap_values_flat,
                base_values=shap_values_selected.base_values,
                data=data_flat,
                feature_names=None  # 稍后用扩展名重新设置
            )
            print(f"  SHAP，展平后shap_values_2d形状: {shap_values_2d.shape}")
        except Exception as e:
            print(f"  SHAP，展平SHAP对象失败: {e}")
            return

        # 9. 生成扩展特征名
        expanded_feature_names = []
        for step in input_steps_to_analyze:
            for feature in input_feature_names:
                forward_h = int(forward_hours - step * batch2_data_time_interval_adjust / 3600)
                expanded_feature_names.append(f"{feature}_-{forward_h}h")

        # 设置扩展特征名
        shap_values_2d.feature_names = expanded_feature_names

        # 10. 创建文件名前缀
        filename_prefix = f"SHAP_{file_title}_+{backward_h}h"
        os.makedirs(output_subfolder, exist_ok=True)  # 确保输出目录存在

        # 11. 保存原始SHAP值==============================================================
        try:
            # 保存SHAP值
            save_path_values = os.path.join(output_subfolder, f"{filename_prefix}_values.csv")
            df_shap_values = pd.DataFrame(shap_values_2d.values, columns=shap_values_2d.feature_names)
            df_shap_values.to_csv(save_path_values, index=False)

            # 保存特征值
            save_path_data = os.path.join(output_subfolder, f"{filename_prefix}_data.csv")
            df_shap_data = pd.DataFrame(shap_values_2d.data, columns=shap_values_2d.feature_names)
            df_shap_data.to_csv(save_path_data, index=False)

            print("  SHAP，完整SHAP值保存完成")
        except Exception as e:
            print(f"  SHAP，保存SHAP值失败: {e}")

        # 12. 创建并保存可视化图==========================================================
        try:
            # 12.1 waterfall
            for i in range(min(4, n_samples)):  # 遍历前4个样本
                shap.plots.waterfall(shap_values_2d[i], max_display=10, show=False)
                plt.savefig(os.path.join(output_subfolder, f"{filename_prefix}_waterfall_sample_{i}.png"))
                plt.close()
            print(f"  SHAP，waterfall图已保存")

            # 12.2 beeswarm
            shap.plots.beeswarm(shap_values_2d, show=False)
            plt.savefig(os.path.join(output_subfolder, f"{filename_prefix}_beeswarm.png"))
            plt.close()
            print(f"  SHAP，beeswarm图已保存")

            # 12.3 bar
            shap.plots.bar(shap_values_2d, show=False)
            plt.savefig(os.path.join(output_subfolder, f"{filename_prefix}_bar.png"))
            plt.close()
            print(f"  SHAP，bar图已保存")

            # 12.4 force_plot
            force_plot = shap.force_plot(
                shap_values_2d.base_values,  # 基准值
                shap_values_2d[:].values,  # SHAP值矩阵
                shap_values_2d[:].data,  # 特征值
                feature_names=shap_values_2d.feature_names,
                matplotlib=False  # 禁用matplotlib渲染
            )
            shap.save_html(os.path.join(output_subfolder, f"{filename_prefix}_force.html"), force_plot)
            print(f"  SHAP，force_plot图已保存")

            # 12.5 heatmap
            shap.plots.heatmap(shap_values_2d[:], show=False)
            plt.savefig(os.path.join(output_subfolder, f"{filename_prefix}_heatmap.png"))
            plt.close()
            print(f"  SHAP，heatmap图已保存")

        except Exception as e:
            print(f"  SHAP，可视化保存失败: {e}")

        # 13. 保存重要特征摘要
        try:
            # 计算平均绝对SHAP值
            mean_abs_shap = np.abs(shap_values_2d.values).mean(axis=0)
            # 创建特征重要性数据框
            feature_importance = pd.DataFrame({
                'Feature': expanded_feature_names,
                'Mean|SHAP|': mean_abs_shap
            })
            # 按重要性排序
            feature_importance = feature_importance.sort_values('Mean|SHAP|', ascending=False)
            # 保存到CSV
            importance_path = os.path.join(output_subfolder, f"{filename_prefix}_feature_importance.csv")
            feature_importance.to_csv(importance_path, index=False)
            print(f"  SHAP，特征重要性已保存: {importance_path}")

        except Exception as e:
            print(f"  SHAP，计算特征重要性失败: {e}")

    print(f"  SHAP，站点 {site_idx} 分析完成")

    return