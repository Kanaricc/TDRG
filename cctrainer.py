from typing import List
from chinopie.modelhelper import TrainHelper
from chinopie.datasets import COCO2014Dataset, get_coco_labels
from chinopie.probes.average_precision_meter import AveragePrecisionMeter
from chinopie.preprocess import DatasetWrapper
from data import MultiScaleCrop
import torch.utils.data
from torch.utils.data import DataLoader
import torch
import torch.nn.utils.clip_grad
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Optimizer
import torchvision
from models import TDRG
import numpy as np
import torchvision.transforms as transforms


def tune_opti(helper: TrainHelper, opti: Optimizer, epochi: int):
    decay = 0.1 if (torch.tensor(helper.get("epoch_step")) == epochi).sum() > 0 else 1.0
    for param_group in opti.param_groups:
        param_group["lr"] = param_group["lr"] * decay


def train(
    epoch_num: int,
    batch_size: int,
    lr: float,
    lrp: float,
    momentum: float,
    weight_decay: float,
    max_clip_grad_norm: float,
    image_size: int,
    epoch_step: List[int] = [40],
):
    helper = TrainHelper(
        "tdrg",
        epoch_num,
        batch_size,
        auto_load_checkpoint=False,
        enable_checkpoint=False,
        checkpoint_save_period=None,
        comment="test",
    )
    helper.register_global_params("epoch_step", epoch_step)
    helper.set_fixed_seed(1)

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

    trainset = COCO2014Dataset(
        helper.get_dataset_slot("coco2014"), lambda x: x, "train"
    )
    trainset, valset = torch.utils.data.random_split(
        trainset,
        [int(len(trainset) * 0.8), len(trainset) - int(len(trainset) * 0.8)],
        generator=torch.Generator().manual_seed(1),
    )
    testset = COCO2014Dataset(helper.get_dataset_slot("coco2014"), transform_val, "val")
    dataloader_train = DataLoader(
        DatasetWrapper(trainset, lambda x: transform_wrapper(x, transform_train)),
        helper.batch_size,
    )
    dataloader_val = DataLoader(
        DatasetWrapper(valset, lambda x: transform_wrapper(x, transform_val)),
        helper.batch_size,
    )
    dataloader_test = DataLoader(testset, helper.batch_size)
    num_classes = trainset[0]["target"].size(-1)
    helper.register_dataset(trainset, dataloader_train, valset, dataloader_val)

    criterion = torch.nn.MultiLabelSoftMarginLoss().to(helper.dev)

    res101 = torchvision.models.resnet101(pretrained=True)
    model = TDRG(res101, num_classes)
    model_parallel = DistributedDataParallel(model).to(helper.dev)

    optimizer = SGD(
        model.get_config_optim(lr, lrp),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    apmeter_train = AveragePrecisionMeter()
    apmeter_val = AveragePrecisionMeter()
    apmeter_test = AveragePrecisionMeter()
    apmeter_train.reset()
    apmeter_val.reset()
    apmeter_test.reset()

    helper.register_probe("map/train")
    helper.register_probe("map/val")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_train/{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_train/k_{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_val/{i}")
    for i in ["OP", "OR", "OF1", "CP", "CR", "CF1"]:
        helper.register_probe(f"other_val/k_{i}")

    helper.ready_to_train()
    for epochi in helper.range_epoch():
        tune_opti(helper, optimizer, epochi)

        # train phase
        model_parallel.train()
        for batchi, data in helper.range_train():
            inputs: Tensor = data["image"]
            targets: Tensor = data["target"]
            inputs = inputs.to(helper.dev)
            targets = inputs.to(helper.dev)

            out_trans, out_gcn, out_sac = model_parallel(inputs)
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
                model_parallel.parameters(), max_norm=max_clip_grad_norm
            )
            optimizer.step()

            helper.update_loss_probe("train", loss, outputs.size(0))
            apmeter_train.add(outputs, targets, data["name"])

        # val phase
        model_parallel.eval()
        for batchi, data in helper.range_val():
            inputs: Tensor = data["image"]
            targets: Tensor = data["target"]
            inputs = inputs.to(helper.dev)
            targets = inputs.to(helper.dev)

            with torch.no_grad():
                out_trans, out_gcn, out_sac = model_parallel(inputs)
            outputs = 0.7 * out_trans + 0.3 * out_gcn

            loss = (
                criterion(outputs, targets)
                + criterion(out_trans, targets)
                + criterion(out_gcn, targets)
                + criterion(out_sac, targets)
            )
            helper.update_loss_probe("val", loss, outputs.size(0))
            apmeter_val.add(outputs, targets, data["name"])

        # log metrics
        ap = apmeter_train.value()
        map = ap.mean()
        OP, OR, OF1, CP, CR, CF1 = apmeter_train.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = apmeter_train.overall_topk(3)

        helper.update_probe("map/train", map)
        helper.tbwriter.add_scalars(
            "ap/train",
            list(zip(get_coco_labels(helper.get_dataset_slot("coco2014")), ap)),
            epochi,
        )
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
        map = ap.mean()
        OP, OR, OF1, CP, CR, CF1 = apmeter_val.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = apmeter_val.overall_topk(3)

        helper.update_probe("map/val", map)
        helper.tbwriter.add_scalars(
            "ap/val",
            list(zip(get_coco_labels(helper.get_dataset_slot("coco2014")), ap)),
            epochi,
        )
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

        helper.end_epoch(map.item())
        apmeter_train.reset()
        apmeter_val.reset()
        apmeter_test.reset()

        if helper.if_need_save_checkpoint():
            torch.save(
                {
                    "model": model_parallel.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "helper": helper.export_state(),
                },
                helper.get_checkpoint_slot(epochi),
            )
        if helper.if_need_save_best_checkpoint():
            torch.save(
                {
                    "model": model_parallel.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "helper": helper.export_state(),
                },
                helper.get_best_checkpoint_slot(),
            )
