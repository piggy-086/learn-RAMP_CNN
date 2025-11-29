# ramp_cnn_2h_tester.py (最终修复版 V6.0 - 引入 Peak 提取)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import sys
import logging

# 确保 data_processor.py 存在且已修正角度归一化
try:
    from data_processor import RadarDataset, RadarDataProcessor, radar_configs, label_map
except ImportError:
    print("FATAL ERROR: 无法导入 data_processor.py。请确保文件存在并包含必要的类和变量。")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


# ==============================================================================
# 优化后的模型定义 (Lite Version) - 保持不变
# ==============================================================================

class Conv3DAutoencoderLite(nn.Module):
    def __init__(self, in_channels: int, output_channels: int = 192):
        super().__init__()
        self.output_channels = output_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, output_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )
        # Decoder 部分省略，因为模型只返回 features
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(output_channels, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.PReLU(),
            nn.ConvTranspose3d(32, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        reconstructed = self.decoder(features)
        return features, reconstructed


class FeatureFusionModuleLite(nn.Module):
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


class RAMP_CNN_Lite(nn.Module):
    def __init__(self, num_classes: int = 6, sequence_length: int = 4):
        super().__init__()
        AE_OUT_CH = 192
        FUSION_OUT_CH = 384

        self.ra_ae = Conv3DAutoencoderLite(in_channels=2, output_channels=AE_OUT_CH)
        self.rv_ae = Conv3DAutoencoderLite(in_channels=1, output_channels=AE_OUT_CH)
        self.va_ae = Conv3DAutoencoderLite(in_channels=1, output_channels=AE_OUT_CH)

        self.fusion = FeatureFusionModuleLite(AE_OUT_CH, AE_OUT_CH, AE_OUT_CH, FUSION_OUT_CH)

        self.output_conv = nn.Sequential(
            nn.Conv3d(FUSION_OUT_CH, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(sequence_length, 128, 128), mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, kernel_size=1)
        )
        self.sequence_length = sequence_length

        # Focal Loss 偏置初始化
        p = 0.01
        bias_init = -np.log((1 - p) / p)

        last_conv = self.output_conv[-1]
        if isinstance(last_conv, nn.Conv3d):
            last_conv.bias.data.fill_(bias_init)
        else:
            logging.warning("Warning: Could not find final Conv3d layer for bias initialization.")

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
# MetricCalculator 类 (关键修改：引入 Peak 提取/NMS)
# ==============================================================================

class MetricCalculator:
    def __init__(self, processor: RadarDataProcessor, score_thresh: float = 0.2,
                 pixel_range_thresh: float = 3.0, pixel_angle_thresh: float = 3.0):
        self.processor = processor
        self.class_indices = list(label_map.keys())

        # 保持极低阈值，但 Peak 提取会大幅减少实际预测数
        self.score_threshold = score_thresh
        self.PIXEL_RANGE_THRESHOLD = pixel_range_thresh
        self.PIXEL_ANGLE_THRESHOLD = pixel_angle_thresh

    def _extract_peaks(self, heatmap_logits: np.ndarray) -> List[Dict]:
        """使用 Max Pooling 提取 CenterNet 风格的 Peak (替代 NMS)"""
        # (C, 1, H, W) -> (C, H, W)
        heatmap = 1.0 / (1.0 + np.exp(-heatmap_logits))
        heatmap_2d = heatmap[:, 0, :, :]

        C, H, W = heatmap_2d.shape

        # 1. 寻找局部最大值 (使用 Max Pooling 模拟 NMS)
        # 将 numpy 数组转换为 torch Tensor
        scores_tensor = torch.from_numpy(heatmap_2d).unsqueeze(0)  # (1, C, H, W)

        # 3x3 Max Pooling，步长 1，填充 1
        # 这将找到每个 3x3 区域内的最大值
        max_pooled = F.max_pool2d(scores_tensor, kernel_size=3, stride=1, padding=1)

        # 2. 只有当分数 = 局部最大值时，才认为它是 Peak
        # 此外，分数必须高于 SCORE_THRESHOLD
        is_peak = (scores_tensor == max_pooled)
        is_above_thresh = (scores_tensor >= self.score_threshold)

        # 结合条件：是 Peak 且高于阈值
        peaks_mask = (is_peak & is_above_thresh).squeeze(0).numpy()  # (C, H, W)

        predictions = []

        for c in range(C):
            class_id = self.class_indices[c]

            # 找到 Peak 的坐标
            y_indices, x_indices = np.where(peaks_mask[c])

            scores_c = heatmap_2d[c]

            for y, x in zip(y_indices, x_indices):
                predictions.append({
                    'class_id': class_id,
                    'score': scores_c[y, x],
                    'y': y,  # 像素 y 坐标 (Range/H)
                    'x': x,  # 像素 x 坐标 (Angle/W)
                })

        return predictions

    def _match_objects(self, preds: List[Dict], gts: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not gts or not preds:
            return np.array([]), np.array([]), np.array([])

        pred_scores = np.array([p['score'] for p in preds])
        pred_tps = np.zeros(len(preds), dtype=bool)
        gt_matched = np.zeros(len(gts), dtype=bool)

        gt_coords = np.array([[g['y'], g['x'], g['class_id']] for g in gts])
        sort_indices = np.argsort(pred_scores)[::-1]

        for p_idx in sort_indices:
            p = preds[p_idx]
            p_y, p_x, p_class = p['y'], p['x'], p['class_id']
            best_match_idx = -1

            for g_idx, g_coord in enumerate(gt_coords):
                g_y, g_x, g_class = g_coord[0], g_coord[1], g_coord[2]

                if gt_matched[g_idx] or g_class != p_class:
                    continue

                # 匹配逻辑：Y->Y, X->X 像素匹配
                y_diff = abs(p_y - g_y)
                x_diff = abs(p_x - g_x)

                if y_diff < self.PIXEL_RANGE_THRESHOLD and x_diff < self.PIXEL_ANGLE_THRESHOLD:
                    best_match_idx = g_idx
                    break

            if best_match_idx != -1:
                pred_tps[p_idx] = True
                gt_matched[best_match_idx] = True

        return pred_tps, pred_scores, np.array([p['class_id'] for p in preds])

    # calculate_ap_ar 保持不变
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

        if len(all_scores) == 0 or num_gt == 0:
            return {'AP': 0.0, 'AR': 0.0, 'num_gt': num_gt}

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
# RAMP_CNNTrainer 类 - 保持不变
# ==============================================================================

class RAMP_CNNTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                 metric_calculator: MetricCalculator):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metric_calculator = metric_calculator
        self.optimizer = None
        self.scheduler = None

        self.criterion = self.centernet_focal_loss

    def centernet_focal_loss(self, pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 4,
                             gamma: float = 2) -> torch.Tensor:
        # 维度转换 (B, C, T, H, W) -> (B, T, H, W, C)
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
            target = target.squeeze()
            target = target.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)

            self.optimizer.zero_grad()
            output = self.model(ra_input, rv_input, va_input)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            data_iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        if loader is None:
            return {'AP': 0.0, 'AR': 0.0, 'num_gt': 0}

        self.model.eval()
        all_predictions_and_gts = []
        all_gt_centers = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                ra_input = batch['ra'].to(self.device)
                rv_input = batch['rv'].to(self.device)
                va_input = batch['va'].to(self.device)
                gt_target = batch['gt']

                output_logits = self.model(ra_input, rv_input, va_input)

                if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
                    logging.error("FATAL: Model output contains NaN/Inf values! Training failed.")
                    return {'AP': 0.0, 'AR': 0.0, 'num_gt': 0}

                gt_target = gt_target.squeeze().cpu().numpy()  # (B, T, H, W, C)
                pred_np = output_logits.cpu().numpy()  # (B, C, T, H, W)

                pred_np_last_frame = pred_np[:, :, -1:, :, :]  # (B, C, 1, H, W)
                gt_np_last_frame = gt_target[:, -1:, :, :, :]  # (B, 1, H, W, C)

                B, C_pred, D_pred, H, W = pred_np_last_frame.shape

                for b in range(B):
                    gt_frame_squeezed = gt_np_last_frame[b, 0, :, :, :]  # (H, W, C)

                    # 提取 GT 目标中心 (像素 y, x)
                    gt_y, gt_x, gt_c = np.where(gt_frame_squeezed > 0.99)

                    current_gts = []
                    for y, x, c_idx in zip(gt_y, gt_x, gt_c):
                        current_gts.append({
                            'class_id': self.metric_calculator.class_indices[c_idx],
                            'y': y,
                            'x': x,
                        })

                    pred_frame_logits = pred_np_last_frame[b, :, 0:1, :, :]
                    current_preds = self.metric_calculator._extract_peaks(
                        pred_frame_logits
                    )

                    # 打印 Logits 的 min/max，用于诊断
                    if b == 0:
                        pred_min = pred_frame_logits.min()
                        pred_max = pred_frame_logits.max()
                        logging.info(f"Batch {b} Max Pred Logit: {pred_max:.4f}, Min Pred Logit: {pred_min:.4f}")
                        logging.info(f"Total predictions found: {len(current_preds)}")

                    all_predictions_and_gts.append((current_preds, current_gts))
                    all_gt_centers.extend(current_gts)

        results = self.metric_calculator.calculate_ap_ar(all_predictions_and_gts, all_gt_centers)
        return results


def plot_metrics(history: Dict[str, List[float]], total_epochs: int):
    # 此方法保持不变
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
        ax2.plot(epochs, history['AP'], label='Average Precision (AP)', color='tab:blue', marker='o')
        ax2.plot(epochs, history['AR'], label='Average Recall (AR)', color='tab:green', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 100)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower left')
    plt.title('RAMP-CNN Training and Evaluation Metrics')
    fig.tight_layout()
    plt.savefig('training_metrics_history_2h.png')
    print("训练指标图已保存为 training_metrics_history_2h.png")


# ==============================================================================
# --- (4) 主运行函数 (test_main - 最终修复配置) ---
# ==============================================================================

def test_main():
    print("\n=========================================================================")
    print("🚀 启动 RAMP-CNN 2小时优化训练测试")
    print("=========================================================================")

    # --- 2小时优化配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TEST_EPOCHS = 15
    TEST_BATCH_SIZE = 16
    TEST_MAX_SEQUENCES = 150
    TEST_NUM_WORKERS = 4
    SEQUENCE_LENGTH = 4

    # 📌 请替换为您的真实数据路径
    data_dir = r"H:\python data\Automotive"

    print(f"使用设备: {device}")
    print(f"测试周期 (TEST_EPOCHS): {TEST_EPOCHS}")
    print(f"最大加载序列数 (TEST_MAX_SEQUENCES): {TEST_MAX_SEQUENCES}")
    print(f"测试 Batch Size: {TEST_BATCH_SIZE}")
    print(f"序列长度 (SEQUENCE_LENGTH): {SEQUENCE_LENGTH} (假设 data_processor.py 支持)")

    try:
        test_dataset = RadarDataset(data_dir, sequence_length=SEQUENCE_LENGTH, max_sequences=TEST_MAX_SEQUENCES)
    except Exception as e:
        print(
            f"FATAL ERROR: 初始化数据集失败，请检查数据路径 '{data_dir}' 及 data_processor.py 中 sequence_length={SEQUENCE_LENGTH} 的兼容性。错误: {e}")
        sys.exit(1)

    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=TEST_NUM_WORKERS,
                             pin_memory=True if torch.cuda.is_available() else False)
    eval_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=TEST_NUM_WORKERS,
                             pin_memory=True if torch.cuda.is_available() else False)

    if len(test_dataset) == 0:
        print("FATAL ERROR: 测试数据集为空。请检查路径和序列设置。")
        sys.exit(1)

    print(f"加载的测试集大小: {len(test_dataset)} 批次: {len(test_loader)}")
    print("-------------------------------------------------------------------------")

    num_classes = len(label_map)
    model = RAMP_CNN_Lite(num_classes=num_classes, sequence_length=SEQUENCE_LENGTH)

    MAX_LR = 2e-4
    BASE_LR = 1e-5

    # --- 性能指标配置 ---
    # SCORE_THRESHOLD 保持极低，但 Peak 提取会控制 FP 数量
    SCORE_THRESHOLD = 1e-6
    PIXEL_THRESHOLD = 3.0

    # --- 3. 初始化评估器，使用 Peak 提取逻辑 ---
    dummy_processor = RadarDataProcessor(radar_configs)

    metric_calculator = MetricCalculator(
        dummy_processor,
        score_thresh=SCORE_THRESHOLD,
        pixel_range_thresh=PIXEL_THRESHOLD,
        pixel_angle_thresh=PIXEL_THRESHOLD
    )

    trainer = RAMP_CNNTrainer(model, test_loader, eval_loader, device, metric_calculator)

    trainer.optimizer = Adam(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    # Cyclic LR 步长设置
    step_size = (len(test_dataset) // TEST_BATCH_SIZE) * 2

    trainer.scheduler = CyclicLR(
        trainer.optimizer,
        base_lr=BASE_LR,
        max_lr=MAX_LR,
        step_size_up=step_size,
        cycle_momentum=False
    )
    training_history = {'train_loss': [], 'AP': [], 'AR': []}

    print(f"\n🚀 开始快速测试，总目标 {TEST_EPOCHS} 个周期。")
    print("=========================================================================")

    for epoch in range(1, TEST_EPOCHS + 1):
        train_loss = trainer.train_epoch(epoch)
        eval_metrics = trainer.evaluate(eval_loader)

        avg_ap = eval_metrics['AP']
        avg_ar = eval_metrics['AR']

        training_history['train_loss'].append(train_loss)
        training_history['AP'].append(avg_ap)
        training_history['AR'].append(avg_ar)

        print(
            f'Epoch {epoch}/{TEST_EPOCHS} | Loss: {train_loss:.6f} | AP: {avg_ap:.2f}%, AR: {avg_ar:.2f}% | Num GT: {eval_metrics["num_gt"]}')
        print(f"当前学习率: {trainer.optimizer.param_groups[0]['lr']:.8f}")
        print("-------------------------------------------------------------------------")

    print("\n✅ 2小时优化测试完成! (请根据实际运行时间调整 MAX_SEQUENCES/EPOCHS)")
    plot_metrics(training_history, len(training_history['train_loss']))


if __name__ == "__main__":
    test_main()