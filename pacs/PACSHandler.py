import torch
import numpy as np
import deeplake
from torchvision import transforms
from torch.utils.data import TensorDataset

class PACSHandler:
    def __init__(self):
        self.pacs_train = deeplake.load("hub://activeloop/pacs-train")   # 8977
        self.pacs_val = deeplake.load("hub://activeloop/pacs-val")       # 1014
        #self.pacs_test = deeplake.load("hub://activeloop/pacs-test")     # 9991
        self.num_classes = 7
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def one_hot_labeling(self, arr):
        one_hot_labels = np.eye(self.num_classes, dtype=np.float32)[arr]
        return torch.tensor(one_hot_labels)
        #return torch.tensor(one_hot_labels, device=self.device)

    def resize_image(self, image, target_size=(224, 224)):
        transform = transforms.Compose([
            transforms.ToTensor(),                     # 227x227x3 -> 3x227x224
            transforms.ToPILImage(),
            transforms.Resize(target_size),            # 3x227x227 -> 3x224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

    def split_domain(self, test_domain):
        datasets = [self.pacs_train, self.pacs_val]
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for dataset in datasets:
            for i, (image, label, domain) in enumerate(zip(dataset.images, dataset.labels, dataset.domains.data()['text'])):
                # Resizing and Normalization
                image_tensor = self.resize_image(image.numpy(), target_size=(224, 224))
                label_tensor = self.one_hot_labeling(label.numpy())

                if test_domain != domain[0]:
                    train_images.append(image_tensor)
                    train_labels.append(label_tensor)
                else:
                    test_images.append(image_tensor)
                    test_labels.append(label_tensor)

        # for i, (image, label, domain) in enumerate(zip(self.pacs_test.images, self.pacs_test.labels, self.pacs_test.domains.data()['text'])):
        #     # Resizing and Normalization
        #     image_tensor = self.resize_image(image.numpy(), target_size=(224, 224))
        #     label_tensor = self.one_hot_labeling(label.numpy())

        #     if test_domain != domain[0]:
        #         train_images.append(image_tensor)
        #         train_labels.append(label_tensor)
        #     else:
        #         test_images.append(image_tensor)
        #         test_labels.append(label_tensor) 

        train_images = torch.stack(train_images)
        train_labels = torch.stack(train_labels)
        train_labels = train_labels.squeeze(dim=1)
        test_images = torch.stack(test_images)
        test_labels = torch.stack(test_labels)
        test_labels = test_labels.squeeze(dim=1)

        print(f"\n==> PACS Handler...")
        print(f"Train Dataset not including {test_domain}")
        print(train_images.shape, train_images.device, train_images.dtype)
        print(train_labels.shape, train_labels.device, train_labels.dtype)
        print(f"\nTest Dataset for {test_domain}")
        print(test_images.shape, test_images.device, test_images.dtype)
        print(test_labels.shape, test_labels.device, test_labels.dtype)

        return TensorDataset(train_images, train_labels), TensorDataset(test_images, test_labels)
