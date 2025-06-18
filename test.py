import torch
# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")
# 检查CUDA是否可用（GPU支持）
print(f"CUDA是否可用: {torch.cuda.is_available()}")
# 如果CUDA可用，输出当前GPU设备信息
if torch.cuda.is_available():
    print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("CUDA不可用，PyTorch将在CPU模式下运行。")








