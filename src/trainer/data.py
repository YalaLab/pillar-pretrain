import copy
import os
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Dict, Iterator

import yaml
from torch.utils.data import DataLoader, DistributedSampler

from .merlin_abd_ct import MerlinAbdCTDataset


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def _resolve_preprocess(preprocess_fn):
    if isinstance(preprocess_fn, (list, tuple)):
        return preprocess_fn[0]
    return preprocess_fn


def get_merlin_abd_ct_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    data_root = getattr(args, "medcsv_data_root", None)

    use_cache = getattr(args, "use_cache", True)
    cache_dir = getattr(args, "cache_dir", None)
    cache_manifest = getattr(args, "cache_manifest", None)

    target_d = getattr(args, "medcsv_target_d", 192)
    target_hw = getattr(args, "target_hw", None)
    pad_value = getattr(args, "medcsv_pad_value", -1.0)
    transform_option = getattr(args, "transform_option", "pad,nearest")

    use_rve = getattr(args, "use_rve", False)
    rve_dir = getattr(args, "rve_dir", None)
    rve_manifest = getattr(args, "rve_manifest", None)

    window_type = getattr(args, "window_type", None)
    if window_type is not None and not hasattr(args, "window_type"):
        args.window_type = window_type

    text_cache_dir = getattr(args, "text_cache_dir", None)
    if text_cache_dir:
        split_suffix = "train" if is_train else "val"
        text_cache_dir = os.path.join(text_cache_dir, split_suffix)

    transforms = _resolve_preprocess(preprocess_fn)
    dataset = MerlinAbdCTDataset(
        input_filename=input_filename,
        transforms=transforms,
        img_paths_key=args.csv_img_key,
        csv_caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        data_root=data_root,
        tokenizer=tokenizer,
        caption_transform=getattr(args, "medcsv_caption_transform", None),
        target_d=target_d,
        target_hw=target_hw,
        pad_value=pad_value,
        transform_option=transform_option,
        use_cache=use_cache,
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        text_cache_dir=text_cache_dir,
        use_rve=use_rve,
        rve_dir=rve_dir,
        rve_manifest=rve_manifest,
        window_type=window_type,
        modality="CT",
        is_train=is_train,
    )

    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size if is_train else args.val_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        timeout=getattr(args, "dataloader_timeout", 300),
        persistent_workers=args.workers > 0,
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_multimodal_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    with open(args.multimodal_config, "r") as f:
        multimodal_datasets = yaml.safe_load(f)

    datasets: Dict[str, DataInfo] = {}
    for dataset_name, dataset_info in multimodal_datasets.items():
        data_path = (
            dataset_info["args"].get("train_data")
            if is_train
            else dataset_info["args"].get("val_data")
        )
        if data_path is None:
            raise ValueError(
                f"Dataset {dataset_name} does not have a path specified in the config."
            )

        dataset_fn = get_dataset_fn(dataset_name)
        args_copy = copy.deepcopy(args)
        for key, value in dataset_info.get("args", {}).items():
            setattr(args_copy, key, value)

        modality_preprocess_fn = None
        if isinstance(preprocess_fn, dict):
            modality_preprocess_fn = preprocess_fn.get(dataset_name, None)
        else:
            modality_preprocess_fn = preprocess_fn

        modality = dataset_info.get("modality")
        if modality is None:
            raise ValueError(
                f"Dataset {dataset_name} does not have a modality specified in the config."
            )

        datasets[modality] = dataset_fn(
            args_copy,
            modality_preprocess_fn,
            is_train=is_train,
            epoch=epoch,
            tokenizer=tokenizer,
        )

    rebuild_on_exhaust = getattr(args, "multimodal_rebuild_on_exhaust", False)
    return MultiDatasetDataloader(
        datasets, rebuild_on_exhaust=rebuild_on_exhaust
    )


def get_dataset_fn(dataset_type):
    if dataset_type == "multimodal":
        return get_multimodal_dataset
    if dataset_type == "merlin_abd_ct":
        return get_merlin_abd_ct_dataset
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    dataset_fn = get_dataset_fn(args.dataset_type)
    data["train"] = dataset_fn(
        args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
    )

    data["val"] = dataset_fn(
        args, preprocess_val, is_train=False, tokenizer=tokenizer
    )

    return data


class MultiDatasetDataloader:
    def __init__(
        self,
        dataloaders: Dict[str, DataInfo],
        rebuild_on_exhaust: bool = False,
    ):
        self.dataloaders = dataloaders
        self.rebuild_on_exhaust = rebuild_on_exhaust
        self.iterators = None
        self._remaining_batches = None
        self.modal_num_samples = {
            key: dl.dataloader.num_samples for key, dl in self.dataloaders.items()
        }
        self.modal_num_batches = {
            key: dl.dataloader.num_batches for key, dl in self.dataloaders.items()
        }

        self.shortest_modal = min(
            self.modal_num_batches, key=self.modal_num_batches.get
        )

        self.longest_modal = max(
            self.modal_num_batches, key=self.modal_num_batches.get
        )
        self._length = self.modal_num_batches[self.longest_modal]

        self.num_samples = self.modal_num_samples[self.longest_modal]
        self.num_batches = self.modal_num_batches[self.longest_modal]

    def __len__(self) -> int:
        return self._length

    def set_remaining_batches(self, remaining_batches: int):
        self._remaining_batches = remaining_batches

    def get_modality_status_at_batch(self, batch_idx: int) -> Dict[str, str]:
        status = {}
        for key in self.modal_num_batches:
            if batch_idx >= self.modal_num_batches[key]:
                status[key] = (
                    f"exhausted (has {self.modal_num_batches[key]} batches)"
                )
            else:
                remaining = self.modal_num_batches[key] - batch_idx
                status[key] = f"active ({remaining} batches remaining)"
        return status

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.exhausted = set()

        if self._remaining_batches is not None:
            batches_to_yield = self._remaining_batches
            completed_batches = self._length - self._remaining_batches

            self.iterators = {}
            for key, dl in self.dataloaders.items():
                if self.modal_num_batches[key] <= completed_batches:
                    self.exhausted.add(key)
                else:
                    self.iterators[key] = iter(dl.dataloader)

            if not self.iterators:
                return

            self._remaining_batches = None
        else:
            self.iterators = {
                key: iter(dl.dataloader) for key, dl in self.dataloaders.items()
            }
            batches_to_yield = self._length

        for _ in range(batches_to_yield):
            combined_batch = {}
            for key, data_iterator in self.iterators.items():
                if key in self.exhausted:
                    continue
                try:
                    batch = next(data_iterator)
                    combined_batch[key] = batch
                except StopIteration:
                    if self.rebuild_on_exhaust:
                        self.iterators[key] = iter(
                            self.dataloaders[key].dataloader
                        )
                        try:
                            batch = next(self.iterators[key])
                            combined_batch[key] = batch
                        except StopIteration:
                            self.exhausted.add(key)
                    else:
                        self.exhausted.add(key)

            if combined_batch:
                yield combined_batch
            else:
                return
