from .plantwild import PlantWild
from .plantdoc import PlantDoc
from .plantvillage import PlantVillage

DATASETS = {
    "plantwild": PlantWild,
    "plantdoc": PlantDoc,
    "plantvillage": PlantVillage,
}


def build_dataset(dataset, root_path, shots):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
    return DATASETS[dataset](root_path, shots)
