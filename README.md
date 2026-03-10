# Weed Detection
**RedEdge** 数据集上进行的杂草检测实验设置与结果。

<img width="378" height="86" alt="image" src="https://github.com/user-attachments/assets/4e003216-3482-4c66-ab5f-0a2193048ec9" />


## 数据集

本次实验使用的数据集路径如下：

/home/cclsol/cxn/Lawin/LWViTs-for-weedmapping/dataset/processed/RedEdge
### 输入通道： RGB + NIR + RE
#### 实验结果1-5：
<img width="470" height="226" alt="image" src="https://github.com/user-attachments/assets/1cb5df63-7051-4f8c-a246-222e1c968fa1" />


##### 实验 1：未加入随机卷积（Random Convolution）。

##### 实验 4：未使用 self.cls_net。

相关代码如下：rgb_rand = self.prorandconv(rgb)

##### 实验 5：使用了融合增强策略。

将随机增强后的特征与原始特征按相同比例进行融合：rgb = 0.5 * rgb_rand + 0.5 * rgb_original

##### 实验六结果：git上代码为实验六与实验七代码

<img width="182" height="118" alt="image" src="https://github.com/user-attachments/assets/71e2e4f1-45ba-4e35-b043-2f01e8017adf" />

黑色为较实验五下降，红色为上升


##### 实验 7

self.cls_net 由 ResNet-18 替换为 ResNet-34。

<img width="340" height="66" alt="image" src="https://github.com/user-attachments/assets/887cc9bb-e053-4965-a5a8-72791a5a3340" />









