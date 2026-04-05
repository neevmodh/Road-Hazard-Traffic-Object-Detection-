import torch
from torch.utils.data import DataLoader

from src.dataset import RoadDataset
from src.transforms import get_transform


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(batch_size=2, train=True):

    dataset = RoadDataset(
        img_dir="/Users/lalu/road_project/data/final/images",
        annotation_file="/Users/lalu/road_project/data/final/annotations.json",
        transforms=get_transform(train=train)
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,   # safe for Mac
        collate_fn=collate_fn
    )

    return data_loader


# ----------- TEST -----------
if __name__ == "__main__":
    loader = get_dataloader()

    images, targets = next(iter(loader))

    print("\n✅ DataLoader Working!")
    print("Batch size:", len(images))
    print("Boxes:", targets[0]["boxes"].shape)
    print("Labels:", targets[0]["labels"])