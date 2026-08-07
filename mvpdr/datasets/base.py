import json
import math
import os
import os.path as osp
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


def read_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def write_json(obj, fpath):
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def read_image(path):
    if not osp.exists(path):
        raise IOError(f"No file exists at {path}")
    return Image.open(path).convert("RGB")


class Datum:
    def __init__(self, impath="", label=0, domain=-1, classname=""):
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    dataset_dir = ""

    def __init__(self, train_x=None, val=None, test=None):
        self._train_x = train_x
        self._val = val
        self._test = test
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = sorted(mapping.keys())
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            return data_sources[0] if len(data_sources) == 1 else data_sources

        print(f"Creating a {num_shots}-shot dataset")
        output = []

        for data_source in data_sources:
            tracker = defaultdict(list)
            for item in data_source:
                tracker[item.label].append(item)

            dataset = []
            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    sampled_items = random.choices(items, k=num_shots) if repeat else items
                dataset.extend(sampled_items)
            output.append(dataset)

        return output[0] if len(output) == 1 else output

    @staticmethod
    def build_split_from_dir(image_dir, classes, train_pct=0.8, val_split=1.0):
        """Build train/val/test split from a directory of class folders.

        Args:
            image_dir: path containing class subdirectories
            classes: list of class directory names
            train_pct: fraction of data for train+val
            val_split: fraction of train portion to use for train (rest goes to val).
                       1.0 means no val split (all train).
        """
        image_names = []
        image_labels = []
        split_codes = []

        for i, cls in enumerate(classes):
            c_root = os.path.join(image_dir, cls)
            names = os.listdir(c_root)
            num = len(names)
            num_trainval = math.ceil(num * train_pct)
            trainval_idx = random.sample(range(num), num_trainval)

            midpoint = math.ceil(len(trainval_idx) * val_split)
            train_idx = set(trainval_idx[:midpoint])
            val_idx = set(trainval_idx[midpoint:])

            c_codes = []
            for j in range(num):
                if j in train_idx:
                    c_codes.append(1)
                elif j in val_idx:
                    c_codes.append(2)
                else:
                    c_codes.append(0)

            split_codes.extend(c_codes)
            for name in names:
                image_names.append(os.path.join(cls, name))
                image_labels.append(i)

        return image_names, image_labels, split_codes

    @staticmethod
    def codes_to_split(split_codes, image_dir, name_list, label_list, classes, join_prefix=True):
        """Convert split codes (0=test, 1=train, 2=val) into lists of Datum objects."""
        splits = {"train": [], "val": [], "test": []}

        for i, code in enumerate(split_codes):
            impath = os.path.join(image_dir, name_list[i]) if join_prefix else name_list[i]
            classname = classes[int(label_list[i])].replace("+", " ")
            item = Datum(impath=impath, label=int(label_list[i]), classname=classname)

            if code == 1:
                splits["train"].append(item)
            elif code == 2:
                splits["val"].append(item)
            else:
                splits["test"].append(item)

        return splits["train"], splits["val"], splits["test"]

    @staticmethod
    def write_split_txt(path, image_dir, name_list, label_list, split_codes):
        """Persist a split to a text file."""
        with open(path, "w") as f:
            for i in range(len(name_list)):
                name = os.path.join(image_dir, name_list[i])
                f.write(f"{name}={label_list[i]}={split_codes[i]}\n")

    @staticmethod
    def read_split_txt(path, classes, join_prefix=None):
        """Read a persisted split text file. If join_prefix is given, prepend it to image paths."""
        splits = {"train": [], "val": [], "test": []}

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().rsplit("=", 2)
                name, label, code = parts
                if join_prefix:
                    name = os.path.join(join_prefix, name)
                classname = classes[int(label)]
                item = Datum(impath=name, label=int(label), classname=classname)

                code = int(code)
                if code == 1:
                    splits["train"].append(item)
                elif code == 2:
                    splits["val"].append(item)
                else:
                    splits["test"].append(item)

        return splits["train"], splits["val"], splits["test"]


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

        interp_mode = T.InterpolationMode.BICUBIC
        self.to_tensor = T.Compose([
            T.Resize(input_size, interpolation=interp_mode),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        img = read_image(item.impath)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.to_tensor(img)

        return img, item.label


def build_data_loader(data_source, batch_size=64, input_size=224, tfm=None, is_train=True, shuffle=False):
    dataset = DatasetWrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
