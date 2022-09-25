import argparse
import os.path as Path

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from model.vanilla_model import Vanilla

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='dataset', help='Dir to the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pth checkpoint file')
    
    opt, _ = parser.parse_known_args()

    assert Path.isfile(opt.model_path), "The pth file does not exist!"

    test_data = datasets.FashionMNIST(
        root="dataset",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    model = Vanilla()

    model.load_state_dict(torch.load(opt.model_path))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()

    cnt = 0
    show_num = 50
    for data in test_data:
        x, y = data[0], data[1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            if (show_num > 0):
                print(f'Predicted: "{predicted}", Actual: "{actual}"')
                show_num -= 1
            if (actual == predicted):
                cnt += 1
    print(f"The accuracy is {cnt / len(test_data)}")