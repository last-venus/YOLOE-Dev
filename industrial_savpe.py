import torch

from ultralytics.data import YOLOConcatDataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.yoloe.train_seg import YOLOESegVPTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, TVPSegmentLoss


class MaskAwareVisualPrompt(LoadVisualPrompt):
    """优先使用实例 mask 生成 visual prompt，只有拿不到 mask 时才退回 bbox。"""

    def __call__(self, labels):
        # 当前样本图像尺寸，格式为 (H, W)
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None

        # 工业/细粒度场景里，mask 通常比 bbox 更能表达目标形状。
        if "masks" in labels and labels["masks"] is not None and len(labels["masks"]):
            masks = labels["masks"]
        elif "bboxes" in labels:
            bboxes = labels["bboxes"]

        # YOLOE 的 visual prompt 仍然需要按类别组织。
        cls = labels["cls"].squeeze(-1).to(torch.int)
        labels["visuals"] = self.get_visuals(cls, imgsz, bboxes=bboxes, masks=masks)
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
    industrial_loss_gains = (1.0, 0.0, 3.0, 1.0, 0.0)

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
        ds.transforms.append(prompt_cls())

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
        dataset = super().build_dataset(img_path, mode, batch)
        return replace_load_visual_prompt(dataset)

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        replace_load_visual_prompt(self.train_loader.dataset)
