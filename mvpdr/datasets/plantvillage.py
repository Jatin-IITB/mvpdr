import os
import random

from .base import DatasetBase

random.seed(1)

TEMPLATE = ["a photo of a {}, a type of plant disease or a healthy leaf."]

CLASSNAME_MAP = [
    "apple scab", "apple black rot", "apple cedar rust", "healthy apple leaf",
    "healthy blueberry leaf", "cherry powdery mildew", "healthy cherry leaf",
    "corn cercospora leaf spot", "corn common rust", "corn northern leaf blight",
    "healthy corn leaf", "grape black rot", "grape black measles",
    "grape isariopsis leaf spot", "healthy grape leaf", "citrus greening",
    "peach bacterial spot", "healthy peach leaf", "bell pepper bacterial spot",
    "healthy bell pepper leaf", "potato early blight", "potato late blight",
    "healthy potato leaf", "healthy raspberry leaf", "healthy soybean leaf",
    "squash powdery mildew", "strawberry leaf scorch", "healthy strawberry leaf",
    "tomato bacterial spot", "tomato early blight", "tomato late blight",
    "tomato leaf mold", "tomato septoria leaf spot", "tomato spider-mites",
    "tomato target spot", "tomato yellow leaf curl virus", "tomato mosaic virus",
    "healthy tomato leaf",
]


class PlantVillage(DatasetBase):
    dataset_dir = "plantvillage"

    def __init__(self, root, num_shots):
        root = os.path.abspath(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.template = TEMPLATE

        self.origin_classes = sorted(os.listdir(self.image_dir))
        self.classes = CLASSNAME_MAP
        split_path = os.path.join(self.dataset_dir, "trainval.txt")

        if not os.path.exists(split_path):
            names, labels, codes = self.build_split_from_dir(self.image_dir, self.origin_classes)
            self.write_split_txt(split_path, self.image_dir, names, labels, codes)
            train, val, test = self.codes_to_split(codes, self.image_dir, names, labels, self.classes)
        else:
            train, val, test = self.read_split_txt(split_path, self.classes)

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        super().__init__(train_x=train, val=val, test=test)
