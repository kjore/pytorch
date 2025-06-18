import torch
import numpy as np

'''初始化张量'''
'''直接从数据初始化张量'''
data = [[1, 2, 3], [4, 5, 6]]
tensor_from_data = torch.tensor(data)
'''从NumPy数组初始化张量'''
numpy_array = np.array(data)
tensor_from_numpy = torch.from_numpy(numpy_array)
'''从标量值初始化张量'''
scalar_value = 3.14
tensor_from_scalar = torch.tensor(scalar_value)
'''从另一个张量初始化张量'''
x_ones = torch.ones_like(tensor_from_data) # 形状相同的全1张量
print(f"x_ones: \n{x_ones}\n")
x_zeros = torch.zeros_like(tensor_from_data) # 形状相同的全0张量
print(f"x_zeros: \n{x_zeros}\n")
x_rand = torch.rand_like(tensor_from_data, dtype=torch.float) # 形状相同的随机张量
print(f"x_rand: \n{x_rand}\n")
'''创建指定形状的张量'''
shape = (2, 3)
tensor_empty = torch.empty(shape)  # 未初始化的张量
print(f"tensor_empty: \n{tensor_empty}\n")
tensor_ones = torch.ones(shape)
print(f"tensor_ones: \n{tensor_ones}\n")
tensor_rands= torch.rand(shape)  # 随机初始化的张量
print(f"tensor_rands: \n{tensor_rands}\n")
# 检查GPU是否可用并设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
# 将张量移动到GPU
tensor_from_data = tensor_from_data.to(device)
# 或者使用以下方式
# tensor_from_data = tensor_from_data.cuda()  # 如果有CUDA可用

'''张量属性'''
print(f"tensor_from_data: \n{tensor_from_data}\n")
print(f"tensor_from_data.shape: {tensor_from_data.shape}")  # 张量的形状
print(f"tensor_from_data.dtype: {tensor_from_data.dtype}")  # 张量的数据类型
print(f"tensor_from_data.device: {tensor_from_data.device}")  # 张量所在的设备


'''张量运算'''
'''索引与切片'''
print(f"tensor_from_data[0, 1]: {tensor_from_data[0, 1]}")  # 访问第0行第1列的元素
print(f"tensor_from_data[:, 1]: {tensor_from_data[:, 1]}")  # 访问第1列的所有元素
print(f"tensor_from_data[0, :]: {tensor_from_data[0, :]}")  # 访问第0行的所有元素
print(f"tensor_from_data[0, 1:3]: {tensor_from_data[0, 1:3]}")  # 访问第0行第1列到第2列的元素
print(f"tensor_from_data[...,-1]: {tensor_from_data[..., -1]}")  # 访问最后一列的所有元素
print(tensor_from_data)
'''张量拼接'''
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])
tensor_concat_dim0 = torch.cat((tensor_a, tensor_b), dim=0)  # 在第0维拼接
print(f"tensor_concat_dim0: \n{tensor_concat_dim0}\n")
tensor_concat_dim1 = torch.cat((tensor_a, tensor_b), dim=1)  # 在第1维拼接
print(f"tensor_concat_dim1: \n{tensor_concat_dim1}\n")
'''张量堆叠'''
''' dim = 0时， c = [ a, b]

    dim =1 时， d = [ [a[0] , b[0] ] , [a[1], b[1] ] ]

    dim = 2 时， e = [     [   [ a[0][0], b[0][0] ]  , [ a[0][1], b[0][1] ]  ,  [ a[0][2],b[0][2] ] ] ,

                         [   [ a[1][0], b[1][0] ]  , [ a[1][1], b[0][1] ]  ,  [ a[1][2],b[1][2] ] ]      ]'''
tensor_stack_dim0 = torch.stack((tensor_a, tensor_b), dim=0)  # 在第0维堆叠
print(f"tensor_stack_dim0: \n{tensor_stack_dim0}\n")
tensor_stack_dim1 = torch.stack((tensor_a, tensor_b), dim=1)  # 在第1维堆叠
print(f"tensor_stack_dim1: \n{tensor_stack_dim1}\n")
tensor_stack_dim2 = torch.stack((tensor_a, tensor_b), dim=2)  # 在第2维堆叠
print(f"tensor_stack_dim2: \n{tensor_stack_dim2}\n")

'''张量运算'''
tensor=torch.ones((3,4))
print(f"tensor: \n{tensor}\n")
tensor_add = tensor + 2  # 张量加法
print(f"tensor_add: \n{tensor_add}\n")
tensor_sub = tensor - 2  # 张量减法
print(f"tensor_sub: \n{tensor_sub}\n")
tensor_mul = tensor * 2  # 张量乘法
print(f"tensor_mul: \n{tensor_mul}\n")
tensor_div = tensor / 2  # 张量除法
print(f"tensor_div: \n{tensor_div}\n")

y1 = tensor @ tensor.T #线性代数中的矩阵乘法
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


print(f"y1: \n{y1}\n")
print(f"y2: \n{y2}\n")
print(f"y3: \n{y3}\n")


z1 = tensor * tensor #对应元素相乘
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"z1: \n{z1}\n")
print(f"z2: \n{z2}\n")
print(f"z3: \n{z3}\n")

'''将单元素张量转化为Python标量'''
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
