# data_processor.py

import numpy as np
import scipy.io as sio
import pandas as pd
import os
import h5py
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import Dataset
import logging
from skimage.transform import resize
import warnings
import sys
import math


#计算高斯核半径

def gaussian_radius(bbox_size: float, min_overlap: float = 0.7) -> int:
    """
    根据目标尺寸估算高斯核的半径 r。
    在 RA 空间中使用经验固定值（通常 CenterNet 使用约 3-5 像素）。
    """
    # 保持固定的经验值
    return 3

#在热图上绘制高斯核峰值
def draw_umich_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
    """
    绘制标准的 CenterNet 风格 2D 高斯核（H, W）。
    Args:
        heatmap: 目标热图 (H, W)。
        center: 目标中心点 (x, y) -> (W, H)。
        radius: 高斯核半径。
    """
    diameter = 2 * radius + 1
    # CenterNet 经验公式：sigma = diameter / 6
    sigma = diameter / 6
    if sigma < 0.1:
        sigma = 0.5

    # 绘制 2D 高斯核
    x_coords = np.arange(diameter)
    y_coords = np.arange(diameter)
    x, y = np.meshgrid(x_coords, y_coords)
    gaussian_2d = np.exp(
        -((x - radius) ** 2 + (y - radius) ** 2 + 1e-6) / (2 * sigma ** 2))  # 加 1e-6 防止除零 (虽然 sigma > 0)

    # 确定要粘贴的区域
    H, W = heatmap.shape[0:2]
    center_x, center_y = center  # (W, H)

    # 计算边界：确保索引不越界
    x_min = max(0, center_x - radius)
    x_max = min(W, center_x + radius + 1)
    y_min = max(0, center_y - radius)
    y_max = min(H, center_y + radius + 1)

    # 计算高斯核对应的边界
    gauss_x_min = radius - (center_x - x_min)
    gauss_x_max = radius + (x_max - center_x)
    gauss_y_min = radius - (center_y - y_min)
    gauss_y_max = radius + (y_max - center_y)

    # 提取区域和高斯核切片
    masked_heatmap = heatmap[y_min:y_max, x_min:x_max]
    masked_gaussian = gaussian_2d[gauss_y_min:gauss_y_max, gauss_x_min:gauss_x_max]

    if masked_heatmap.shape == masked_gaussian.shape:
        # 使用 Max 操作进行叠加，防止覆盖已有的高斯峰值
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    else:
        logging.warning(
            f"Gaussian mask shape mismatch! Target: {masked_heatmap.shape}, Gaussian: {masked_gaussian.shape}. Skipping GT draw.")

    return heatmap


#可视化检查

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
warnings.filterwarnings("ignore", category=np.ComplexWarning)

# 雷达配置参数
radar_configs: Dict[str, Any] = {
    "startFreqConst_GHz": 77.0,
    "bandwidth_GHz": 0.67,
    "chirpDuration_usec": 60.0,
    "freqSlopeConst_MHz_usec": 21.0,
    "numAdcSamples": 128,
    "digOutSampleRate": 4000.0,
    "numLoops": 255,
    "framePeriodicity_msec": 33.33333,
}

# 类别映射
label_map: Dict[int, str] = {
    0: 'person', 2: 'car', 3: 'motorbike',
    5: 'bus', 7: 'truck', 80: 'cyclist',
}

# 标签文件的列名，根据数据手册确定
LABEL_COLUMNS = ['uid', 'class', 'px', 'py', 'wid', 'len']
MIN_LABEL_FILE_SIZE = 100


class RadarDataProcessor:
    def __init__(self, radar_config: Dict):
        self.config = radar_config
        self.MAX_RANGE_FILTER = 50.0
        self.setup_parameters()

    def setup_parameters(self):
        self.start_freq = self.config["startFreqConst_GHz"] * 1e9
        self.bandwidth = self.config["bandwidth_GHz"] * 1e9
        self.chirp_duration = self.config["chirpDuration_usec"] * 1e-6
        self.slope = self.config["freqSlopeConst_MHz_usec"] * 1e12
        self.num_samples = self.config["numAdcSamples"]
        self.sample_rate = self.config["digOutSampleRate"] * 1e3
        self.num_chirps = self.config["numLoops"]
        self.num_rx = 4
        self.num_tx = 2
        self.c = 3e8
        self.max_range = (self.sample_rate * self.c) / (2 * self.slope)

    def load_radar_frame(self, mat_file: str) -> np.ndarray:
        try:
            mat_contents = sio.loadmat(mat_file)
            data_key = next((k for k, v in mat_contents.items()
                             if isinstance(v, np.ndarray) and v.ndim >= 2 and v.dtype != object and
                             v.size >= self.num_samples * self.num_chirps), None)

            if data_key is None: raise ValueError("No suitable data array found in MAT file using scipy.io.")
            data = mat_contents[data_key]
        except Exception:
            try:
                with h5py.File(mat_file, 'r') as f:
                    data_key = next((k for k in f.keys() if isinstance(f[k], h5py.Dataset)), None)
                    if data_key is None: raise ValueError("No suitable dataset found in MAT file using h5py.")
                    data = f[data_key][:]
            except Exception as e_h5py:
                # 降级为 warning，确保程序继续
                logging.warning(f"Could not load radar data from {mat_file}: {e_h5py}")
                return None

        try:
            expected_total_dims = self.num_samples * self.num_chirps * self.num_rx * self.num_tx
            if data.size != expected_total_dims: return None

            data = data.flatten().reshape(self.num_samples, self.num_chirps, self.num_rx, self.num_tx, order='F')
            if not np.iscomplexobj(data): data = data.astype(np.complex64)

            return data.astype(np.complex64)
        except Exception as e:
            logging.error(f"Error reshaping or processing data from {mat_file}: {e}")
            return None

    def range_fft(self, adc_data: np.ndarray) -> np.ndarray:
        return np.fft.rfft(adc_data, axis=0)

    def doppler_fft(self, range_fft: np.ndarray) -> np.ndarray:
        doppler_fft = np.fft.fft(range_fft, axis=1)
        return np.fft.fftshift(doppler_fft, axes=1)

    def angle_fft(self, doppler_fft: np.ndarray) -> np.ndarray:
        return np.fft.fft(doppler_fft, axis=-1)

    def preprocess_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if heatmap.ndim != 2: raise ValueError("Heatmap must be 2D for preprocessing.")
        log_heatmap = 10 * np.log10(heatmap + 1e-6)
        min_val = np.min(log_heatmap)
        max_val = np.max(log_heatmap)
        if max_val == min_val:
            normalized_heatmap = np.zeros_like(log_heatmap)
        else:
            normalized_heatmap = (log_heatmap - min_val) / (max_val - min_val)
        resized_heatmap = resize(
            normalized_heatmap,
            target_size,
            anti_aliasing=True
        ).astype(np.float32)
        return resized_heatmap

    def generate_rva_heatmap(self, adc_data: np.ndarray) -> Dict[str, np.ndarray]:
        # 展平 Rx/Tx 维度以创建虚拟天线
        adc_data_virtual = adc_data.reshape(self.num_samples, self.num_chirps, self.num_rx * self.num_tx)
        range_profile = self.range_fft(adc_data_virtual)
        doppler_profile = self.doppler_fft(range_profile)
        rva_cube = self.angle_fft(doppler_profile)
        abs_rva_cube = np.abs(rva_cube).astype(np.float32)

        heatmaps = {
            'ra_mag': np.mean(abs_rva_cube, axis=1),  # (R, A)
            'rv_mag': np.mean(abs_rva_cube, axis=2),  # (R, V)
            'va_mag': np.mean(abs_rva_cube, axis=0),  # (V, A)
        }
        return heatmaps

    def load_labels(self, label_file: str) -> pd.DataFrame:
        """加载标签文件（强制类型检查和日志）"""
        empty_df = pd.DataFrame(columns=LABEL_COLUMNS)
        if not os.path.exists(label_file) or os.path.getsize(label_file) < MIN_LABEL_FILE_SIZE:
            return empty_df
        try:
            labels = pd.read_csv(
                label_file,
                header=None,
                names=LABEL_COLUMNS,
                dtype={'class': 'Int64', 'px': 'float', 'py': 'float'},  # 使用 Int64 避免 NaN
                on_bad_lines='skip'
            )
            labels = labels.dropna(subset=['class', 'px', 'py'])
            labels['class'] = labels['class'].astype(int)

        except Exception as e:
            logging.error(f"❌ Error loading label file {label_file}: {e}")
            return empty_df
        return labels

    def cartesian_to_polar(self, x: float, y: float) -> Tuple[float, float]:
        """笛卡尔坐标转极坐标 (范围, 角度)"""
        range_val = np.sqrt(x ** 2 + y ** 2)
        # 注意：这里使用 atan2(x, y) 对应于 x 轴（侧向）和 y 轴（距离）
        angle_val = np.arctan2(x, y)
        return range_val, angle_val

    def create_ground_truth_heatmap(self, labels: pd.DataFrame, heatmap_shape: Tuple) -> np.ndarray:
        """使用高斯核创建 CenterNet 风格的 ground truth 热图"""
        _, H, W, num_classes = heatmap_shape
        gt_heatmap = np.zeros(heatmap_shape, dtype=np.float32)

        if labels.empty:
            return gt_heatmap

        class_to_index = {cls_id: i for i, cls_id in enumerate(label_map.keys())}
        radius = gaussian_radius(1.0)

        for _, label in labels.iterrows():
            try:
                class_id = int(label['class'])
                px, py = float(label['px']), float(label['py'])
            except (ValueError, KeyError, TypeError) as e:
                logging.warning(f"Label conversion error for row: {label.to_dict()}. Error: {e}")
                continue

            if class_id not in class_to_index: continue

            range_val, angle_val = self.cartesian_to_polar(px, py)
            class_index = class_to_index[class_id]

            # Range 过滤保持不变
            if range_val > self.MAX_RANGE_FILTER or range_val < 0.1: continue

            # --- 关键修正 ---

            # Y轴 (Range/距离) - 归一化到 [0, H]
            # range_bin_y 是像素 y 坐标 (行索引)
            range_bin_y = int(np.clip(range_val / self.max_range * H, 0, H - 1))

            # X轴 (Angle/角度) - 归一化到 [0, W]
            # 假设 RA 图的物理角度范围是 [-pi/2, pi/2]
            # angle_bin_x 是像素 x 坐标 (列索引)
            # 修正公式: (angle + pi/2) / pi * W
            angle_bin_x = int(np.clip((angle_val + np.pi / 2.0) / np.pi * W, 0, W - 1))

            # --------------------

            # 使用高斯核绘制 GT
            heatmap_slice = gt_heatmap[0, :, :, class_index]  # 形状 (H, W)
            # CenterNet 约定 center=(x, y) -> (W, H)。这里的 x 是 angle_bin_x，y 是 range_bin_y
            draw_umich_gaussian(heatmap_slice, (angle_bin_x, range_bin_y), radius)

        return gt_heatmap


class RadarDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 8, transform=None, max_sequences: int = None):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.processor = RadarDataProcessor(radar_configs)
        self.max_sequences = max_sequences
        self.num_classes = len(label_map)

        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self) -> List[Dict]:
        sequences = []

        # 修正：提前设置 max_sequences 限制，避免扫描全部数据
        max_limit = self.max_sequences if self.max_sequences is not None else float('inf')

        for entry in os.scandir(self.data_dir):
            if not entry.is_dir(): continue
            date_path = entry.path
            radar_folder = os.path.join(date_path, "radar_raw_frame")
            label_folder = os.path.join(date_path, "text_labels")

            if not (os.path.exists(radar_folder) and os.path.exists(label_folder)): continue

            radar_files = sorted([f for f in os.listdir(radar_folder) if f.endswith('.mat')])

            for i in range(0, len(radar_files) - self.sequence_length + 1):
                if len(sequences) >= max_limit:
                    return sequences  # 达到限制，立即返回

                current_radar_files = [os.path.join(radar_folder, f)
                                       for f in radar_files[i:i + self.sequence_length]]
                label_paths = []
                valid_labels_found_in_sequence = False

                # 检查序列中的每一个标签文件
                for radar_file in radar_files[i:i + self.sequence_length]:
                    try:
                        base_name_mat = os.path.splitext(radar_file)[0]
                        file_index = int(base_name_mat)
                    except ValueError:
                        continue

                    base_name_csv = f"{file_index:010d}"

                    label_filename = base_name_csv + '.csv'
                    label_path = os.path.join(label_folder, label_filename)

                    label_paths.append(label_path)

                    # 仅检查最后一帧是否有有效标签 (用于 CenterNet，我们只检测最后一帧)
                    if radar_file == radar_files[i + self.sequence_length - 1]:
                        is_valid = os.path.exists(label_path) and os.path.getsize(label_path) > MIN_LABEL_FILE_SIZE
                        if is_valid: valid_labels_found_in_sequence = True

                # 关键检查：确保序列长度正确且最后一帧有标签
                if len(label_paths) == self.sequence_length and valid_labels_found_in_sequence:
                    sequence = {
                        'radar_files': current_radar_files,
                        'label_files': label_paths
                    }
                    sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        ra_sequence, rv_sequence, va_sequence, gt_sequence = [], [], [], []
        seq_len = self.sequence_length
        H, W = 128, 128
        num_classes = len(label_map)
        RA_TARGET_SIZE = (H, W)
        RV_TARGET_SIZE = (H, W)
        VA_TARGET_SIZE = (H, W)

        # 零填充帧 (用于数据缺失时)
        ra_fill = np.zeros((2, H, W), dtype=np.float32)
        rv_fill = np.zeros((H, W), dtype=np.float32)
        va_fill = np.zeros((H, W), dtype=np.float32)
        gt_fill_shape = (1, H, W, num_classes)
        gt_fill = np.zeros(gt_fill_shape, dtype=np.float32)

        for i, (radar_file, label_file) in enumerate(zip(sequence_info['radar_files'], sequence_info['label_files'])):

            adc_data = self.processor.load_radar_frame(radar_file)

            # 填充缺失数据
            if adc_data is None:
                ra_sequence.append(ra_fill)
                rv_sequence.append(rv_fill)
                va_sequence.append(va_fill)
                gt_sequence.append(gt_fill)
                continue

            heatmaps = self.processor.generate_rva_heatmap(adc_data)

            # 只有最后一帧需要生成 GT (CenterNet 目标检测通常只在最后一帧进行)
            if i == seq_len - 1:
                labels = self.processor.load_labels(label_file)
                gt_heatmap = self.processor.create_ground_truth_heatmap(labels, gt_fill_shape)
            else:
                gt_heatmap = gt_fill  # 其它帧的 GT 设为零

            # 处理 RVA 热图
            ra_mag_processed = self.processor.preprocess_heatmap(heatmaps['ra_mag'], RA_TARGET_SIZE)
            # 原始 RAMP-CNN 使用 2 通道 RA 输入 (可能是 Mag 和 Phase，这里使用两次 Mag)
            ra_2channel_frame = np.stack([ra_mag_processed, ra_mag_processed], axis=0)

            rv_processed = self.processor.preprocess_heatmap(heatmaps['rv_mag'], RV_TARGET_SIZE)
            va_processed = self.processor.preprocess_heatmap(heatmaps['va_mag'], VA_TARGET_SIZE)

            ra_sequence.append(ra_2channel_frame)
            rv_sequence.append(rv_processed)
            va_sequence.append(va_processed)
            gt_sequence.append(gt_heatmap)

        try:
            # 堆叠序列数据
            ra_stacked = np.stack(ra_sequence, axis=1)  # (C_in=2, T, H, W)
            rv_stacked = np.stack(rv_sequence, axis=0)  # (T, H, W)
            va_stacked = np.stack(va_sequence, axis=0)  # (T, H, W)
            gt_stacked = np.stack(gt_sequence, axis=0)  # (T, 1, H, W, C)

            # 确保类型正确，并强制复制内存
            return {
                'ra': torch.from_numpy(ra_stacked).float().contiguous(),
                'rv': torch.from_numpy(rv_stacked).float().contiguous(),
                'va': torch.from_numpy(va_stacked).float().contiguous(),
                # 使用 .copy() 保证 Numpy 数组到 Tensor 的内存安全
                'gt': torch.from_numpy(gt_stacked.copy()).float().contiguous()
            }
        except Exception as e:
            logging.error(f"FATAL: Error stacking sequence data at index {idx}: {e}. Returning zero tensors.",
                          exc_info=True)
            # 发生错误时返回全零 Tensor，避免程序崩溃
            return {
                'ra': torch.zeros((2, seq_len, H, W), dtype=torch.float32),
                'rv': torch.zeros((seq_len, H, W), dtype=torch.float32),
                'va': torch.zeros((seq_len, H, W), dtype=torch.float32),
                'gt': torch.zeros((seq_len, 1, H, W, num_classes), dtype=torch.float32),
            }


if __name__ == "__main__":
    # --- 示例测试代码 ---
    data_dir = r"H:\python data\Automotive"
    # 测试 max_sequences 限制
    dataset = RadarDataset(data_dir, sequence_length=8, max_sequences=1)

    print(f"Total sequences found: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        # 检查序列的最后一帧，即有标签的那一帧 (索引 seq_len - 1)
        gt_slice = sample['gt'][-1, 0, :, :, :]
        max_score = gt_slice.max().item()
        num_pixels_gt = torch.sum(gt_slice > 0.01).item()

        print(f"RA Input Shape: {sample['ra'].shape}")
        print(f"RV Input Shape: {sample['rv'].shape}")
        print(f"GT Target Shape: {sample['gt'].shape}")
        print(f"✅ Max GT Score (应为 1.0): {max_score}")
        print(f"✅ Total GT Pixels (>0.01) in Sample (应 > 目标数量): {num_pixels_gt}")
    else:
        print("❌ 未找到任何有效的序列。")
    print("data_processor.py (最终高斯核版) 优化测试完成.")