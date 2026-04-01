import torch

def Fv6_safe_torch_load(file_path, map_location=None):
    """
    安全加载 PyTorch 模型，使用 weights_only=True 如果可用
    """
    try:
        # PyTorch 1.13+ 支持 weights_only
        if hasattr(torch, 'load') and hasattr(torch.load, '__call__'):
            return torch.load(file_path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError) as e:
        # 如果出错，回退到传统方法并显示警告
        print(f"安全加载失败 (PyTorch 1.13+?): {e}")

    print(f"警告: 使用潜在不安全的加载方式: {file_path}")
    print(f"建议升级到 PyTorch 1.13+")
    return torch.load(file_path, map_location=map_location)