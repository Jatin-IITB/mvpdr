import os
import random

from .base import DatasetBase

random.seed(1)

TEMPLATE = ["a photo of a {}, a type of plant disease."]


class PlantWild(DatasetBase):
    dataset_dir = "plantwild"

    def __init__(self, root, num_shots):
        root = os.path.abspath(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.template = TEMPLATE

        self.classes = sorted(os.listdir(self.image_dir))
        split_path = os.path.join(self.dataset_dir, "trainval.txt")

        if not os.path.exists(split_path):
            names, labels, codes = self.build_split_from_dir(
                self.image_dir, self.classes, val_split=0.875
            )
            self.write_split_txt(split_path, self.image_dir, names, labels, codes)
            train, val, test = self.codes_to_split(codes, self.image_dir, names, labels, self.classes)
        else:
            train, val, test = self.read_split_txt(split_path, self.classes, join_prefix=self.image_dir)

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)
