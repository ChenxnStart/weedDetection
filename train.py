# -*- coding: utf-8 -*-
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import torchvision.utils as vutils

# --- 1. 路径修复 (防止 ModuleNotFoundError) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 引入之前的 Dataset 和 Model ---
# 确保文件名和类名与我们之前修好的一致
from data.rededge_dataset import EschikonDataset 
from MODEL.model_UNet_Side_SE_Random import MSFusionUNet as MSFusionModel

# ==========================================
# 📊 1. 评估工具类 (计算 mIoU 和 F1)
# ==========================================
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixel_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def mean_intersection_over_union(self):
        miou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0) - 
            np.diag(self.confusion_matrix) + 1e-6
        )
        return np.nanmean(miou), miou

    def f1_score(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
        return np.nanmean(f1), f1

    def add_batch(self, gt_image, pre_image):
        # ⚠️ 关键修改：只处理 0, 1, 2 的像素，忽略 255
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        self.confusion_matrix += count.reshape(self.num_class, self.num_class)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

# ==========================================
# 📝 2. 日志系统
# ==========================================
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'train_log_{timestamp}.txt')
    
    logger = logging.getLogger("MSUNet_Train")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ==========================================
# 🚀 3. 模型包装 (适配输入参数)
# ==========================================
class MSFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

# ==========================================
# 🔥 4. 训练主程序
# ==========================================
def train():
    # --- A. 配置参数 (直接写在这里防止找不到 yaml) ---
    cfg = {
        'data_root': "/home/cclsol/cxn/Lawin/LWViTs-for-weedmapping/dataset/processed",
        'train_split': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/MSU-Net/code/splits/train.txt",
        'val_split': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/MSU-Net/code/splits/test.txt", # 确认文件名
        'num_classes': 3,
        'batch_size': 4,       # 显存不够改 2
        'lr': 0.0001,
        'epochs': 100,
        'save_dir': os.path.join(project_root, "checkpoints", "Eschikon_Run_Random_clsnet")
    }

    # --- B. 初始化 ---
    logger = setup_logger(cfg['save_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 启动训练 | 设备: {device}")

    # --- C. 数据集 ---
    logger.info("🔄 加载数据集...")
    train_dataset = EschikonDataset(cfg['data_root'], cfg['train_split'])
    val_dataset = EschikonDataset(cfg['data_root'], cfg['val_split'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    
    logger.info(f"✅ 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # --- D. 模型 ---
    # 5通道: RGB(3) + NIR(1) + RE(1)
    model = MSFusion(in_channels=5, num_classes=cfg['num_classes']).to(device)

    # --- E. 损失函数 (关键修复) ---
    # 权重: [背景, 作物, 杂草] -> 重点关注杂草
    weights = torch.tensor([1.0, 3.0, 8.0]).to(device)
    # ignore_index=255: 忽略之前导致报错的边缘黑边
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    evaluator = Evaluator(cfg['num_classes'])

    best_f1 = 0.0

    # --- F. 循环 ---
   # ... 在循环内部 ...
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")

        check_first_batch = True

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            # Mask 清洗
            masks[(masks > 2) & (masks != 255)] = 255

            outputs, outputs_a, rgb, rgb_rand = model(images)

            if check_first_batch:
                unique_vals = torch.unique(masks)
                print(f"\n🔍 [DEBUG] 当前 Batch Mask 包含数值: {unique_vals.cpu().tolist()}")

                vis = torch.cat([rgb[:4], rgb_rand[:4]], dim=0).detach().cpu()
                vutils.save_image(
                    vis,
                    os.path.join(cfg['save_dir'], f"randconv_vis_epoch_{epoch+1}.png"),
                    normalize=True
                )

                check_first_batch = False

            loss_main = criterion(outputs, masks)
            loss_aug = criterion(outputs_a, masks)
            loss = 0.5 * loss_main + 0.5 * loss_aug

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        # --- 验证 ---
        model.eval()
        val_loss = 0
        evaluator.reset()
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images, masks = images.to(device), masks.to(device)
                
                # 同样需要清洗验证集 Mask
                masks[(masks > 2) & (masks != 255)] = 255

                # outputs = model(images)
                # loss = criterion(outputs, masks)
                # val_loss += loss.item()

                # preds = torch.argmax(outputs, dim=1)
                outputs, outputs_a, rgb, rgb_rand = model(images)

                loss_main = criterion(outputs, masks)
                loss_aug = criterion(outputs_a, masks)
                loss = 0.5 * loss_main + 0.5 * loss_aug
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                evaluator.add_batch(masks.cpu().numpy(), preds.cpu().numpy())

        # --- 指标计算 ---
        mIoU, class_iou = evaluator.mean_intersection_over_union()
        mF1, class_f1 = evaluator.f1_score()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # 日志输出
        # Class 0:背景, 1:作物, 2:杂草
        log_msg = (
            f"\n📊 Ep {epoch+1} Result:\n"
            f"   Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}\n"
            f"   mIoU: {mIoU:.2%} | mF1: {mF1:.2%}\n"
            f"   [F1 Detail] Crop: {class_f1[1]:.2%} | Weed: {class_f1[2]:.2%} (目标)"
        )
        logger.info(log_msg)

        # 保存最优模型 (依据 杂草F1 或 mF1)
        # 这里我设为依据 mF1，你也可以改成 class_f1[2]
        if mF1 > best_f1:
            best_f1 = mF1
            save_path = os.path.join(cfg['save_dir'], "best_model.pth")
            torch.save(model.model.state_dict(), save_path)
            logger.info(f"🌟 新高! 模型已保存: {save_path}")

if __name__ == '__main__':
    train()
