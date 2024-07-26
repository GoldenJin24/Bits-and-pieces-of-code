# Bits and pieces of code
 学习中遇到的一些零散代码，每次想用只是记起来之前用过（问过GPT），但是就是不知道该怎么写
## 生物信息学相关
- python下读取fasta文件,想要字典类型的话修改一下就行
```
#g观察需不需要把U换成T
sequences = []
seq_ids = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split(" ")[0]
            seq_ids.append(seq_id)
            sequences.append("")
        else:
            sequence = line.strip().replace("U","T")
            sequences[-1] += sequence

id_seq_dict = {}
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            seq_id = line[1:].strip().split(" ")[0]
            id_seq_dict[seq_id] = ""
        else:
            sequence = line.strip().replace("U","T")
            id_seq_dict[seq_id] += sequence
```
## 机器学习相关
- 评估模型二分类效果，包括AUC，accuracy等
![alt text](<~[)1(`)[7_7MZF%X5{KK039.png>)
```
# y_pred_proba为两列；test_data_label为一列,直接从测试集取label这一列
def model_eval(y_pred_proba,test_data_label):
    import pandas as pd  
    import numpy as np  
    import matplotlib.pyplot as plt  
    from sklearn.metrics import roc_curve, auc,confusion_matrix, accuracy_score    

    # 1.ROC曲线
    # 观察y_pred_proba是不是只取一列
    df = pd.concat([y_pred_proba.iloc[:,1], test_data_label], axis=1)  
    # 计算ROC曲线  
    fpr, tpr, thresholds = roc_curve(test_data_label, y_pred_proba.iloc[:,1])  
    # 计算AUC  
    roc_auc = auc(fpr, tpr)  
    # 绘制ROC曲线  
    plt.figure()  
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 随机猜测的线  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.0])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver Operating Characteristic (ROC) Curve')  
    plt.legend(loc='lower right')  
    plt.grid()  
    plt.show()

    # 2.混淆矩阵的四个值
    threshold = 0.5  
    y_pred = (y_pred_proba.iloc[:,1] >= threshold).astype(int)  
    test_data_label = test_data_label.squeeze().tolist()
    cm = confusion_matrix(test_data_label, y_pred)  
    accuracy = accuracy_score(test_data_label, y_pred)  
    TP = cm[1, 1]  # 真正类  
    FN = cm[1, 0]  # 假负类  
    FP = cm[0, 1]  # 假正类  
    TN = cm[0, 0]  # 真负类  
    # 计算精确率、召回率和F1分数  
    precision =TP / (TP + FP)  if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  

    #每一行是actual 0 和 1；每一列是pred 0 和 1
    print("准确率:", accuracy)
    print(f'Confusion Matrix:\n{cm}')  
    print(f'Precision: {precision:.4f}')  
    print(f'Recall: {recall:.4f}')  
    print(f'F1 Score: {f1:.4f}') 
```
- Dataframe的label替换，lamba方法
```
df['label'] = df['label'].apply(lambda x: 1 if x == 'Healthy Control' else 0)  
```


## 深度学习相关
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