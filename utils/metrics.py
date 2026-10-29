import torch
import segmentation_models_pytorch as smp


class BinarySegmentationMetrics:
    """累加整个 epoch 的像素统计量，再使用 SMP 计算指标。"""

    def __init__(self, device, threshold=0.5):
        self.threshold = threshold
        # 顺序为 TP、FP、FN、TN，每一项的形状都是 [1, 1]。
        self.totals = torch.zeros((4, 1, 1), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, logits, masks):
        probabilities = torch.sigmoid(logits.detach())
        stats = smp.metrics.get_stats(
            probabilities,
            masks.long(),
            mode="binary",
            threshold=self.threshold,
        )
        # get_stats 返回 [B, 1]；先按样本求和，避免对 batch 指标直接平均。
        self.totals += torch.stack(
            [value.sum(dim=0, keepdim=True) for value in stats]
        )

    def compute(self):
        tp, fp, fn, tn = self.totals
        # 已经累计成 [1, 1]，无需再次归约；None 会应用 SMP 的零分母处理。
        options = {"reduction": None, "zero_division": 0.0}
        foreground_iou = smp.metrics.iou_score(tp, fp, fn, tn, **options)
        # 把背景当作正类时，TP/FP/FN/TN 分别变为 TN/FN/FP/TP。
        background_iou = smp.metrics.iou_score(tn, fn, fp, tp, **options)
        return {
            "iou": foreground_iou.item(),
            "miou": ((foreground_iou + background_iou) / 2).item(),
            "dice": smp.metrics.f1_score(tp, fp, fn, tn, **options).item(),
            "accuracy": smp.metrics.accuracy(tp, fp, fn, tn, **options).item(),
            "precision": smp.metrics.precision(tp, fp, fn, tn, **options).item(),
            "recall": smp.metrics.recall(tp, fp, fn, tn, **options).item(),
        }
