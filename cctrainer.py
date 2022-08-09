from typing import List
from chinopie.modelhelper import TrainHelper
from chinopie.datasets import COCO2014Dataset, get_coco_labels
from chinopie.probes.average_precision_meter import AveragePrecisionMeter
from chinopie.preprocess import DatasetWrapper
from data import MultiScaleCrop
from data.partialcoco import COCO2014Partial
import torch.utils.data
from torch.utils.data import DataLoader
import torch
import torch.nn.utils.clip_grad
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Optimizer
import torchvision
from models import TDRG
from models.baseline import Baseline
import numpy as np
import torchvision.transforms as transforms
from loguru import logger
from losses import get_loss

import inspect


def tune_opti(helper: TrainHelper, opti: Optimizer, epochi: int):
    decay = 0.1 if (torch.tensor(helper.get("epoch_step")) == epochi).sum() > 0 else 1.0
    for param_group in opti.param_groups:
        param_group["lr"] = param_group["lr"] * decay


def train(
    loss_type: str = "partial_bce",
    epoch_num: int = 50,
    batch_size: int = 8,
    lr: float = 0.01,
    lrp: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    max_clip_grad_norm: float = 10.0,
    image_size: int = 448,
    epoch_step: List[int] = [40],
    label_percent: float = 0.1,
    comment: str = "tdrg-pbce(0.01)-label(0.1)",
    dry_run=False,
):
    helper = TrainHelper(
        "tdrg",
        epoch_num,
        batch_size,
        auto_load_checkpoint=True,
        enable_checkpoint=True,
        checkpoint_save_period=None,
        comment=comment,
        dry_run=dry_run,
    )
    helper.register_global_params("epoch_step", epoch_step)

    transform_train = transforms.Compose(
        [
            transforms.Resize((image_size + 64, image_size + 64)),
            MultiScaleCrop(
                image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def transform_wrapper(x, transform):
        x["image"] = transform(x["image"])
        return x

    if loss_type == "bce":
        raw_trainset = COCO2014Partial(
            helper.get_dataset_slot("coco2014"), lambda x: x, "train"
        )
    else:
        raw_trainset = COCO2014Partial(
            helper.get_dataset_slot("coco2014"),
            lambda x: x,
            "train",
            negatives_as_neg1=True,
        )
        logger.info("use negative 1 for negative labels")
    # seems not proper. we do not need to drop validation set.
    raw_trainset.drop_labels(label_percent, 1)
    trainset, valset = torch.utils.data.random_split(
        raw_trainset,
        [
            int(len(raw_trainset) * 0.8),
            len(raw_trainset) - int(len(raw_trainset) * 0.8),
        ],
        generator=torch.Generator().manual_seed(1),
    )
    testset = COCO2014Partial(helper.get_dataset_slot("coco2014"), transform_val, "val")
    dataloader_train = DataLoader(
        DatasetWrapper(trainset, lambda x: transform_wrapper(x, transform_train)),
        helper.batch_size,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        DatasetWrapper(valset, lambda x: transform_wrapper(x, transform_val)),
        helper.batch_size,
        drop_last=True,
    )
    dataloader_test = DataLoader(testset, helper.batch_size, drop_last=True)
    num_classes = trainset[0]["target"].size(-1)
    helper.register_dataset(trainset, dataloader_train, valset, dataloader_val)
    helper.register_test_dataset(testset, dataloader_test)

    criterion = get_loss(loss_type)

    helper.set_fixed_seed(1)
    # model = Baseline(2048,num_classes).to(helper.dev)
    model = TDRG(torchvision.models.resnet101(pretrained=True), num_classes).to(
        helper.dev
    )

    optimizer = SGD(
        model.get_config_optim(lr, lrp),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    if helper.if_need_load_checkpoint():
        checkpoint_path = helper.find_latest_checkpoint()
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            helper.load_from_checkpoint(checkpoint["helper"])

    apmeter_train = AveragePrecisionMeter(False)
    apmeter_val = AveragePrecisionMeter(False)
    apmeter_train.reset()
    apmeter_val.reset()

    helper.register_probe("map/train")
    helper.register_probe("map/val")
    helper.register_probe("map/test")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_train/{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_train/k_{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_val/{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_val/k_{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_test/{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_test/k_{i}")

    helper.ready_to_train()
    for epochi in helper.range_epoch():
        tune_opti(helper, optimizer, epochi)

        # train phase
        model.train()
        for batchi, data in helper.range_train():
            inputs: Tensor = data["image"]
            targets: Tensor = data["target"]
            inputs = inputs.to(helper.dev)
            targets = targets.to(helper.dev)

            out_trans, out_gcn, out_sac = model(inputs)
            outputs = 0.7 * out_trans + 0.3 * out_gcn

            loss = (
                criterion(outputs, targets)
                + criterion(out_trans, targets)
                + criterion(out_gcn, targets)
                + criterion(out_sac, targets)
            )
            helper.validate_loss(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), max_norm=max_clip_grad_norm
            )
            optimizer.step()

            helper.update_loss_probe("train", loss, outputs.size(0))
            apmeter_train.add(outputs, targets, data["name"])

        # val phase
        model.eval()
        for batchi, data in helper.range_val():
            inputs: Tensor = data["image"]
            targets: Tensor = data["target"]
            inputs = inputs.to(helper.dev)
            targets = targets.to(helper.dev)

            # to get full labels for ap calculation
            gt_targets = raw_trainset.get_full_labels(data["index"]).to(helper.dev)
            gt_targets[gt_targets == -1] = 0

            with torch.no_grad():
                out_trans, out_gcn, out_sac = model(inputs)
            outputs = 0.7 * out_trans + 0.3 * out_gcn
            if batchi == 0:
                logger.debug(f"model output: {outputs}")

            loss = (
                criterion(outputs, targets)
                + criterion(out_trans, targets)
                + criterion(out_gcn, targets)
                + criterion(out_sac, targets)
            )

            helper.update_loss_probe("val", loss, outputs.size(0))
            apmeter_val.add(outputs, gt_targets, data["name"])

        # log metrics
        ap = apmeter_train.value()
        train_map = ap.mean()
        OP, OR, OF1, CP, CR, CF1 = apmeter_train.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = apmeter_train.overall_topk(3)

        helper.update_probe("map/train", train_map)
        for k, v in {
            "OP": OP,
            "OR": OR,
            "OF1": OF1,
            "CP": CP,
            "CR": CR,
            "CF1": CF1,
        }.items():
            helper.update_probe(f"other_train/{k}", v)

        for k, v in {
            "OP": OP_k,
            "OR": OR_k,
            "OF1": OF1_k,
            "CP": CP_k,
            "CR": CR_k,
            "CF1": CF1_k,
        }.items():
            helper.update_probe(f"other_train/k_{k}", v)

        ap = apmeter_val.value()
        val_map = ap.mean()
        OP, OR, OF1, CP, CR, CF1 = apmeter_val.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = apmeter_val.overall_topk(3)

        helper.update_probe("map/val", val_map)
        for k, v in {
            "OP": OP,
            "OR": OR,
            "OF1": OF1,
            "CP": CP,
            "CR": CR,
            "CF1": CF1,
        }.items():
            helper.update_probe(f"other_val/{k}", v)

        for k, v in {
            "OP": OP_k,
            "OR": OR_k,
            "OF1": OF1_k,
            "CP": CP_k,
            "CR": CR_k,
            "CF1": CF1_k,
        }.items():
            helper.update_probe(f"other_val/k_{k}", v)

        helper.end_trainval(train_map.item(), val_map.item())
        apmeter_train.reset()
        apmeter_val.reset()

        if helper.if_need_save_checkpoint():
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "helper": helper.export_state(),
                },
                helper.get_checkpoint_slot(epochi),
            )
        if helper.if_need_save_best_checkpoint():
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "helper": helper.export_state(),
                },
                helper.get_best_checkpoint_slot(),
            )

        # test phase
        if helper.if_need_run_test_phase():
            model.eval()
            apmeter_test = AveragePrecisionMeter(False)
            for batchi, data in helper.range_test():
                inputs: Tensor = data["image"]
                targets: Tensor = data["target"]
                inputs = inputs.to(helper.dev)
                targets = targets.to(helper.dev)

                with torch.no_grad():
                    out_trans, out_gcn, out_sac = model(inputs)
                outputs = 0.7 * out_trans + 0.3 * out_gcn
                loss = (
                    criterion(outputs, targets)
                    + criterion(out_trans, targets)
                    + criterion(out_gcn, targets)
                    + criterion(out_sac, targets)
                )

                helper.update_loss_probe("test", loss, outputs.size(0))
                apmeter_test.add(outputs, targets, data["name"])

            ap = apmeter_test.value()
            map = ap.mean()
            OP, OR, OF1, CP, CR, CF1 = apmeter_test.overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = apmeter_test.overall_topk(3)

            helper.update_probe("map/test", map)
            helper.update_test_score(map.item())
            for k, v in {
                "OP": OP,
                "OR": OR,
                "OF1": OF1,
                "CP": CP,
                "CR": CR,
                "CF1": CF1,
            }.items():
                helper.update_probe(f"other_test/{k}", v)

            for k, v in {
                "OP": OP_k,
                "OR": OR_k,
                "OF1": OF1_k,
                "CP": CP_k,
                "CR": CR_k,
                "CF1": CF1_k,
            }.items():
                helper.update_probe(f"other_test/k_{k}", v)
        helper.end_epoch()


if __name__ == "__main__":
    TrainHelper.auto_bind_and_run(train)
