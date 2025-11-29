# train.py (RAMP-CNN 引入 FP16)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
# 🔑 关键修改 2: 引入 AMP
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


# 导入数据处理模块和配置
try:
    from data_processor import RadarDataset, RadarDataProcessor, radar_configs, label_map
except ImportError:
    logging.error("FATAL ERROR: 无法导入 data_processor.py。请确保文件存在且内容正确。")
    sys.exit(1)


# ==============================================================================
# 模型定义 (RAMP_CNN 架构)
# ==============================================================================

class Conv3DAutoencoder(nn.Module):
    def __init__(self, in_channels: int, output_channels: int = 256):
        super().__init__()
        self.output_channels = output_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.PReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.ConvTranspose3d(64, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        reconstructed = self.decoder(features)
        return features, reconstructed


class FeatureFusionModule(nn.Module):
    def __init__(self, ra_channels: int, rv_channels: int, va_channels: int, output_channels: int):
        super().__init__()
        self.rv_to_ra = nn.Sequential(
            nn.Conv3d(rv_channels, ra_channels, kernel_size=1),
            nn.BatchNorm3d(ra_channels),
            nn.ReLU(inplace=True)
        )
        self.va_to_ra = nn.Sequential(
            nn.Conv3d(va_channels, ra_channels, kernel_size=1),
            nn.BatchNorm3d(ra_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(ra_channels * 3, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, ra_features: torch.Tensor, rv_features: torch.Tensor, va_features: torch.Tensor) -> torch.Tensor:
        rv_proj = self.rv_to_ra(rv_features)
        va_proj = self.va_to_ra(va_features)
        target_size = ra_features.shape[2:]
        rv_aligned = F.interpolate(rv_proj, size=target_size, mode='trilinear', align_corners=False)
        va_aligned = F.interpolate(va_proj, size=target_size, mode='trilinear', align_corners=False)
        fused = torch.cat([ra_features, rv_aligned, va_aligned], dim=1)
        fused = self.fusion_conv(fused)
        return fused


class RAMP_CNN(nn.Module):
    def __init__(self, num_classes: int = 6, sequence_length: int = 8):
        super().__init__()
        self.ra_ae = Conv3DAutoencoder(in_channels=2, output_channels=256)
        self.rv_ae = Conv3DAutoencoder(in_channels=1, output_channels=256)
        self.va_ae = Conv3DAutoencoder(in_channels=1, output_channels=256)
        self.fusion = FeatureFusionModule(256, 256, 256, 512)
        self.output_conv = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(sequence_length, 128, 128), mode='trilinear', align_corners=False),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, kernel_size=1),
        )

    def forward(self, ra_input: torch.Tensor, rv_input: torch.Tensor, va_input: torch.Tensor) -> torch.Tensor:
        ra_features, _ = self.ra_ae(ra_input)
        rv_input_3d = rv_input.unsqueeze(1)
        rv_features, _ = self.rv_ae(rv_input_3d)
        va_input_3d = va_input.unsqueeze(1)
        va_features, _ = self.va_ae(va_input_3d)
        fused_features = self.fusion(ra_features, rv_features, va_features)
        output = self.output_conv(fused_features)
        return output


# ==============================================================================
# AP/AR 评估指标计算器  (评估仍使用 FP32 确保精度)
# ==============================================================================

class MetricCalculator:
    def __init__(self, processor: RadarDataProcessor):
        self.processor = processor
        self.class_indices = list(label_map.keys())
        self.score_threshold = 0.2
        self.RANGE_THRESHOLD = 1.0
        self.ANGLE_THRESHOLD = 0.1

    def _extract_peaks(self, heatmap_logits: np.ndarray) -> List[Dict]:
        """使用 PyTorch Max Pooling 替代低效的 Numpy 3x3 循环进行 Peak 提取。"""
        C, D, H, W = heatmap_logits.shape
        heatmap_logits = heatmap_logits.reshape(C, H, W)

        # 转换为 torch Tensor 并进行 Sigmoid
        # 确保使用 .float() 以维持评估精度
        scores_tensor = torch.from_numpy(heatmap_logits).float().unsqueeze(0)  # (1, C, H, W)
        scores_tensor = torch.sigmoid(scores_tensor)

        # 3x3 Max Pooling
        max_pooled = F.max_pool2d(scores_tensor, kernel_size=3, stride=1, padding=1)

        # 只有当分数 = 局部最大值且高于 SCORE_THRESHOLD 时，才认为是 Peak
        is_peak = (scores_tensor == max_pooled)
        is_above_thresh = (scores_tensor >= self.score_threshold)
        peaks_mask = (is_peak & is_above_thresh).squeeze(0).numpy()  # (C, H, W)

        predictions = []
        heatmap_np = scores_tensor.squeeze(0).numpy()

        for c in range(C):
            class_id = self.class_indices[c]
            y_indices, x_indices = np.where(peaks_mask[c])

            for y, x in zip(y_indices, x_indices):
                score = heatmap_np[c, y, x]

                # 转换回物理世界坐标 (用于匹配)
                range_val = ((y + 0.5) / H) * self.processor.max_range
                angle_val = ((x + 0.5) / W) * 2 * np.pi - np.pi

                predictions.append({
                    'class_id': class_id,
                    'score': score,
                    'range': range_val,
                    'angle': angle_val,
                })

        return predictions

    def _match_objects(self, preds: List[Dict], gts: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not gts or not preds:
            return np.array([]), np.array([]), np.array([])

        pred_scores = np.array([p['score'] for p in preds])
        pred_tps = np.zeros(len(preds), dtype=bool)
        gt_matched = np.zeros(len(gts), dtype=bool)

        gt_coords = np.array([[g['range'], g['angle'], g['class_id']] for g in gts])
        sort_indices = np.argsort(pred_scores)[::-1]

        for p_idx in sort_indices:
            p = preds[p_idx]
            p_range, p_angle, p_class = p['range'], p['angle'], p['class_id']
            best_match_idx = -1

            for g_idx, g_coord in enumerate(gt_coords):
                g_range, g_angle, g_class = g_coord[0], g_coord[1], g_coord[2]

                if gt_matched[g_idx] or g_class != p_class:
                    continue

                range_diff = abs(p_range - g_range)
                angle_diff = abs(p_angle - g_angle)

                if range_diff < self.RANGE_THRESHOLD and angle_diff < self.ANGLE_THRESHOLD:
                    best_match_idx = g_idx
                    break

            if best_match_idx != -1:
                pred_tps[p_idx] = True
                gt_matched[best_match_idx] = True

        return pred_tps, pred_scores, np.array([p['class_id'] for p in preds])

    def calculate_ap_ar(self, all_preds: List[Tuple], all_gts: List[Dict]) -> Dict:
        all_tp = []
        all_scores = []
        num_gt = len(all_gts)

        for preds, gts in all_preds:
            tps, scores, _ = self._match_objects(preds, gts)
            all_tp.append(tps)
            all_scores.append(scores)

        if all_tp:
            all_tp = np.concatenate(all_tp).astype(bool)
        else:
            all_tp = np.array([], dtype=bool)
        all_scores = np.concatenate(all_scores) if all_scores else np.array([])

        if len(all_scores) == 0 or num_gt == 0: return {'AP': 0.0, 'AR': 0.0, 'num_gt': num_gt}

        sort_indices = np.argsort(all_scores)[::-1]
        all_tp = all_tp[sort_indices]

        tp_cumsum = np.cumsum(all_tp).astype(float)
        fp_cumsum = np.cumsum(~all_tp).astype(float)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_gt + 1e-6)

        recall = np.concatenate(([0.], recall))
        precision = np.concatenate(([1.], precision))
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        i = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
        ar = recall[-1] if num_gt > 0 else 0.0

        return {'AP': ap * 100, 'AR': ar * 100, 'num_gt': num_gt}


# ==============================================================================
# RAMP_CNNTrainer 类 (引入 GradScaler)
# ==============================================================================

class RAMP_CNNTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                 metric_calculator: MetricCalculator, scaler: GradScaler):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metric_calculator = metric_calculator
        self.optimizer = None
        self.scheduler = None
        self.criterion = self.centernet_focal_loss
        self.best_ap = 0.0
        # 引入 GradScaler
        self.scaler = scaler

    def centernet_focal_loss(self, pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 4,
                             gamma: float = 2) -> torch.Tensor:
        pred_logits = pred_logits.permute(0, 2, 3, 4, 1).contiguous()
        target = target.permute(0, 2, 3, 4, 1).contiguous()
        pred_prob = torch.sigmoid(pred_logits)
        pos_weight = target * torch.pow(target, alpha)
        neg_weight = (1. - target) * torch.pow(1. - target, alpha)
        pos_loss = -torch.log(pred_prob.clamp(min=1e-4)) * torch.pow(1. - pred_prob, gamma)
        neg_loss = -torch.log(1. - pred_prob.clamp(max=1.0 - 1e-4)) * torch.pow(pred_prob, gamma)
        loss = pos_weight * pos_loss + neg_weight * neg_loss
        num_targets = torch.sum(target.gt(0.99).float())
        loss = torch.sum(loss) / torch.clamp(num_targets, min=1.0)
        return loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        data_iterator = tqdm(self.train_loader, desc=f'Epoch {epoch} (Train)', leave=False)

        for _, batch in enumerate(data_iterator):
            ra_input = batch['ra'].to(self.device)
            rv_input = batch['rv'].to(self.device)
            va_input = batch['va'].to(self.device)

            target = batch['gt'].to(self.device)
            target = target.squeeze(2).permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

            self.optimizer.zero_grad()

            # 使用 autocast 启用 FP16
            with autocast():
                output = self.model(ra_input, rv_input, va_input)
                loss = self.criterion(output, target)

            # 使用 GradScaler 进行反向传播和优化
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # CyclicLR 调度器应在优化器更新后调用
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            data_iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss


    def save_checkpoint(self, epoch: int, ap_score: float, is_best: bool, history: Dict):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_ap': self.best_ap,
            'train_loss_history': history['train_loss'],
            'AP_history': history['AP'],
            'AR_history': history['AR'],
            'scaler_state_dict': self.scaler.state_dict(),  # 存储 GradScaler 状态
        }
        torch.save(state, f'ramp_cnn_checkpoint_latest.pth')
        if is_best:
            torch.save(state, f'ramp_cnn_checkpoint_best_ap_{ap_score:.4f}.pth')
            logging.info(f"💾 NEW BEST Checkpoint saved with AP: {ap_score:.4f}%")

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        if loader is None: return {'AP': 0.0, 'AR': 0.0, 'num_gt': 0}
        self.model.eval()
        all_predictions_and_gts = []
        all_gt_centers = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                ra_input = batch['ra'].to(self.device)
                rv_input = batch['rv'].to(self.device)
                va_input = batch['va'].to(self.device)
                gt_target = batch['gt']

                # 评估时也使用 autocast，但不需要 GradScaler
                with autocast():
                    output_logits = self.model(ra_input, rv_input, va_input)

                pred_np = output_logits.cpu().float().numpy()[:, :, -1:, :, :]  # 转换回 float
                gt_np = gt_target.cpu().numpy()[:, -1:, :, :, :, :]

                B, C_pred, D_pred, H, W = pred_np.shape
                processor = self.metric_calculator.processor
                MAX_RANGE = processor.max_range

                for b in range(B):
                    gt_frame = gt_np[b, 0, :, :, :, :]
                    gt_frame_squeezed = gt_frame.squeeze(0)
                    gt_y, gt_x, gt_c = np.where(gt_frame_squeezed > 0.99)

                    current_gts = []
                    for y, x, c_idx in zip(gt_y, gt_x, gt_c):
                        range_val = ((y + 0.5) / H) * MAX_RANGE
                        angle_val = ((x + 0.5) / W) * 2 * np.pi - np.pi
                        if range_val > 0.1:
                            current_gts.append({
                                'class_id': self.metric_calculator.class_indices[c_idx],
                                'range': range_val,
                                'angle': angle_val
                            })

                    pred_frame_logits = pred_np[b, :, 0:1, :, :]
                    current_preds = self.metric_calculator._extract_peaks(
                        pred_frame_logits
                    )

                    all_predictions_and_gts.append((current_preds, current_gts))
                    all_gt_centers.extend(current_gts)

        results = self.metric_calculator.calculate_ap_ar(all_predictions_and_gts, all_gt_centers)
        return results


# ==============================================================================
# main 函数 (改进 CyclicLR)
# ==============================================================================

def main():
    # --- 训练配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = r"H:\python data\Automotive"


    TOTAL_EPOCHS = 50
    BATCH_SIZE = 4
    # 设定 CyclicLR 周期 (5个 Epoch)
    CYC_EPOCHS = 5

    MAX_SEQUENCES_TO_LOAD = None
    NUM_WORKERS = 4
    SEQUENCE_LENGTH = 8
    TRAIN_RATIO = 0.8

    if 'cuda' not in device.type:
        logging.warning("Warning: CUDA not available. Running on CPU may be extremely slow.")
        # 如果是 CPU，Batch Size 必须是 1 且 num_workers 必须是 0
        if BATCH_SIZE > 1: BATCH_SIZE = 1
        if NUM_WORKERS > 0: NUM_WORKERS = 0

    print(f"使用设备: {device}")
    print(f"Batch Size: {BATCH_SIZE} (降低以节省显存)")
    print(f"CyclicLR 周期: {CYC_EPOCHS} Epochs")

    # --- 数据集加载和划分 ---
    full_dataset = RadarDataset(data_dir, sequence_length=SEQUENCE_LENGTH, max_sequences=MAX_SEQUENCES_TO_LOAD)
    if len(full_dataset) == 0:
        logging.error("FATAL ERROR: 数据集为空。")
        return

    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False
    )
    eval_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"训练集大小: {len(train_subset)} | 验证集大小: {len(val_subset)}")
    print("-------------------------------------------------------------------------")

    # --- 模型、优化器和 Trainer 初始化 ---
    num_classes = len(label_map)
    model = RAMP_CNN(num_classes=num_classes, sequence_length=SEQUENCE_LENGTH)

    #  初始化 GradScaler
    scaler = GradScaler()

    dummy_processor = RadarDataProcessor(radar_configs)
    metric_calculator = MetricCalculator(dummy_processor)
    # 传入 scaler
    trainer = RAMP_CNNTrainer(model, train_loader, eval_loader, device, metric_calculator, scaler)

    MAX_LR = 2e-4
    BASE_LR = 1e-5

    trainer.optimizer = Adam(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    # 调整 CyclicLR 步长
    iters_per_epoch = len(train_subset) // BATCH_SIZE
    step_size_up = max(1, iters_per_epoch * CYC_EPOCHS)

    trainer.scheduler = CyclicLR(
        trainer.optimizer, base_lr=BASE_LR, max_lr=MAX_LR,
        step_size_up=step_size_up, cycle_momentum=False
    )

    training_history = {'train_loss': [], 'AP': [], 'AR': []}

    print(f"\n🚀 开始训练，总目标 {TOTAL_EPOCHS} 个周期。")
    print(f"LR将在 {CYC_EPOCHS} Epoch ({step_size_up} steps) 内循环一次。")
    print("=========================================================================")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # 训练
        train_loss = trainer.train_epoch(epoch)
        # 评估 (在验证集上)
        eval_metrics = trainer.evaluate(eval_loader)

        avg_ap = eval_metrics['AP']
        is_best = avg_ap > trainer.best_ap
        if is_best:
            trainer.best_ap = avg_ap

        training_history['train_loss'].append(train_loss)
        training_history['AP'].append(avg_ap)
        training_history['AR'].append(eval_metrics['AR'])

        print(
            f'Epoch {epoch}/{TOTAL_EPOCHS} | Loss: {train_loss:.6f} | Val AP: {avg_ap:.2f}%, Val AR: {eval_metrics["AR"]:.2f}% | Num GT: {eval_metrics["num_gt"]}')

        # 保存 Checkpoint
        trainer.save_checkpoint(epoch, avg_ap, is_best, training_history)
        print("-------------------------------------------------------------------------")

    print("\n✅ 训练完成!")
    plot_metrics(training_history, len(training_history['train_loss']))


def plot_metrics(history: Dict[str, List[float]], total_epochs: int):
    epochs = range(1, total_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Focal)', color=color)
    ax1.plot(epochs, history['train_loss'], label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--')

    if 'AP' in history and 'AR' in history:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Performance (%)', color=color)
        ax2.plot(epochs, history['AP'], label='Validation AP', color='tab:blue', marker='o')
        ax2.plot(epochs, history['AR'], label='Validation AR', color='tab:green', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 100)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower left')

    plt.title('RAMP-CNN Training and Evaluation Metrics')
    fig.tight_layout()
    plt.savefig('training_metrics_history.png')
    print("训练指标图已保存为 training_metrics_history.png")


if __name__ == "__main__":
    main()