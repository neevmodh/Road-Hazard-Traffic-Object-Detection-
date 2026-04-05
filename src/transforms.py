import torchvision.transforms as T

def get_transform(train=True):
    transforms = []

    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)