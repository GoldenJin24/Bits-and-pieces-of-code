# Bits and pieces of code
 学习中遇到的一些零散代码，每次想用只是记起来之前用过（问过GPT），但是就是不知道该怎么写
- pytorch中观察每一层的输入输出形状变化
```
#默认self.XXX才会纳入，
#而自带的类似于x.view()等函数不会认为是一个layer
import torch
from model import CNNClassifier
def get_shape(module, input, output):
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

model = CNNClassifier()

# 注册forward hook来获取每一层的输入和输出形状变化
for name, layer in model.named_children():
    layer.register_forward_hook(get_shape)

# 创建一个随机输入进行测试
# 注意input.shape 要自己思考 （b，4，L）
input_tensor = torch.randn(1, 4, 50)

# 将输入传递给模型
output = model(input_tensor)
```
- 递归消除list内部又有一个list嵌套，展开它成为单个元素
```
#嵌套的列表全部拼接在一起，将列表内部的列表全部打开，要求单个元素
def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

nested_list = [[1, 2, [3, 4]], [5, [6, 7]], 8, [9]]
flattened_list = flatten_list(nested_list)
print(flattened_list)

#或者要求输入的是列表
def flatten_and_merge(*args):
    def flatten_list(nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(flatten_list(item))
            else:
                result.append(item)
        return result

    merged_list = []
    for lst in args:
        merged_list.extend(lst)

    flattened_list = flatten_list(merged_list)
    return flattened_list
# 四个列表
list1 = [1, [2, 3], 4]
list2 = [5, [6, 7]]
list3 = [[8, 9], 10]
list4 = [11, [12, 13]]

result = flatten_and_merge(list1, list2, list3, list4)
print(result)
```