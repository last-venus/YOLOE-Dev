import torch

from ultralytics.data import YOLOConcatDataset, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegVPTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, TVPSegmentLoss
from ultralytics.utils.torch_utils import unwrap_model


class MaskAwareVisualPrompt(LoadVisualPrompt):
    """优先使用实例 mask 生成 visual prompt，只有拿不到 mask 时才退回 bbox。"""

    def __init__(self, nc, scale_factor=1 / 8):
        super().__init__(scale_factor=scale_factor)
        self.nc = nc

    def __call__(self, labels):
        # 当前样本图像尺寸，格式为 (H, W)
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None

        # 工业/细粒度场景里，mask 通常比 bbox 更能表达目标形状。
        if "masks" in labels and labels["masks"] is not None and len(labels["masks"]):
            masks = labels["masks"]
        elif "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * torch.tensor(imgsz)[[1, 0, 1, 0]]  # denormalize boxes

        # 训练和验证统一使用全局类语义：
        # visuals 的通道号始终等于数据集全局类 id，visuals_cls 仅记录当前图里实际出现的类别。
        cls = labels["cls"].squeeze(-1).to(torch.int)
        cls_global = torch.unique(cls, sorted=True)
        labels["visuals_cls"] = cls_global.to(labels["cls"].dtype)
        local_visuals = self.get_visuals(cls, imgsz, bboxes=bboxes, masks=masks)
        visuals = local_visuals.new_zeros((self.nc, *local_visuals.shape[1:]))
        if cls_global.numel():
            visuals[cls_global] = local_visuals
        labels["visuals"] = visuals
        return labels


class IndustrialTVPSegmentLoss(TVPSegmentLoss):
    """工业场景版 VP loss。

    默认 YOLOE 的 VP segmentation loss 更偏向只优化分类项；
    这里保留 visual-prompt 路径，但把完整的 segmentation loss 一起回传。
    """

    def __init__(self, model, tal_topk=10, tal_topk2=None, loss_gains=(1.0, 1.0, 1.0, 1.0, 0.0)):
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_gains = loss_gains

    def loss(self, preds, batch):
        # 还没切到 VP 分支时，直接返回 0 loss，避免错误监督。
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(5, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        loss_vec, loss_items = self.vp_criterion(preds, batch)

        # loss_gains 用于控制各项 loss 的参与程度。
        # 当前顺序与 v8SegmentationLoss 保持一致：
        # (box, seg, cls, dfl, sem)
        gains = torch.tensor(self.loss_gains, device=loss_vec.device, dtype=loss_vec.dtype)
        weighted = loss_vec * gains
        return weighted, loss_items * gains


class IndustrialYOLOESegModel(YOLOESegModel):
    """只在 visual-prompt 训练时替换 loss，其余行为保持 YOLOE 原样。"""

    # overfit12 场景下，先弱化 seg loss，通常更容易先把类别和定位学稳。
    # 顺序为：(box, seg, cls, dfl, sem)
    industrial_loss_gains = (1.0, 1.0, 1.5, 1.0, 0.0)

    def loss(self, batch, preds=None):
        visual_prompt = batch.get("visuals", None) is not None
        if not visual_prompt:
            return super().loss(batch, preds)

        # 只在 VP 训练时切换到工业版 criterion。
        if (
            not hasattr(self, "criterion")
            or getattr(self, "_criterion_visual_prompt", None) != visual_prompt
            or getattr(self, "_criterion_type", None) != "industrial_vp"
        ):
            self.criterion = (
                E2ELoss(self, IndustrialTVPSegmentLoss)
                if getattr(self, "end2end", False)
                else IndustrialTVPSegmentLoss(self, loss_gains=self.industrial_loss_gains)
            )
            self._criterion_visual_prompt = visual_prompt
            self._criterion_type = "industrial_vp"

        if preds is None:
            # VP 训练时只传 visual prompt，不走 text prompt。
            preds = self.forward(
                batch["img"],
                tpe=None,
                vpe=batch.get("visuals", None),
            )
        return self.criterion(preds, batch)


def replace_load_visual_prompt(dataset, prompt_cls=MaskAwareVisualPrompt):
    """把数据集里的默认 LoadVisualPrompt 替换成自定义的 mask-aware 版本。"""

    def _replace_prompt(ds):
        ds.transforms.transforms = [t for t in ds.transforms.transforms if not isinstance(t, LoadVisualPrompt)]
        ds.transforms.append(prompt_cls(ds.data["nc"]))

    if isinstance(dataset, YOLOConcatDataset):
        for ds in dataset.datasets:
            _replace_prompt(ds)
    else:
        _replace_prompt(dataset)
    return dataset


class IndustrialSAVPETrainer(YOLOESegVPTrainer):
    """工业场景的小样本 SAVPE trainer。

    设计目标：
    1. 保留 SAVPE 的 visual prompt 能力
    2. 训练时回传完整 VP segmentation loss，提升 box / mask 的可优化性
    3. 优先用实例 mask 生成 visual prompt，更适合工业场景的细粒度目标
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = IndustrialYOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        else:
            img_paths = img_path if isinstance(img_path, list) else [img_path]
            datasets = [
                build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], mode=mode, stride=gs)
                for im_path in img_paths
            ]
            dataset = YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        return replace_load_visual_prompt(dataset)

    @staticmethod
    def _share_batch_visuals(batch: dict[str, torch.Tensor]) -> None:
        """Use prompts from other samples in the same batch as extra negative visual prompts.

        单类图多图训练时，单张图通常只会激活一个 visual prompt 通道。
        这里把同一 batch 里其他样本的 prompt 拷到对应全局类通道，形成一个小型 prompt bank，
        让每张图都能同时看到真实的正类和负类 prompt，而不是只和全零通道对比。
        """
        visuals = batch.get("visuals", None)
        if visuals is None or visuals.ndim != 4 or visuals.shape[0] < 2:
            return

        prompt_mask = visuals.flatten(2).abs().sum(-1) > 0  # (B, C)
        available_classes = torch.where(prompt_mask.any(0))[0]
        if not len(available_classes):
            return

        shared_visuals = visuals.clone()
        for cls_idx in available_classes.tolist():
            source_idx = torch.where(prompt_mask[:, cls_idx])[0][0]
            missing = ~prompt_mask[:, cls_idx]
            if missing.any():
                shared_visuals[missing, cls_idx] = shared_visuals[source_idx, cls_idx]
        batch["visuals"] = shared_visuals

    def preprocess_batch(self, batch):
        """工业 SAVPE 训练只使用视觉 prompt，不再依赖 text features。"""
        batch = DetectionTrainer.preprocess_batch(self, batch)
        self._share_batch_visuals(batch)
        return batch

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        replace_load_visual_prompt(self.train_loader.dataset)
