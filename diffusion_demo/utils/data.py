import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

CIFAR10_CLASSES_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR10_CLASSES_PROMPTS = [
    'a photo of an airplane',
    'a photo of an automobile',
    'a photo of a bird',
    'a photo of a cat',
    'a photo of a deer',
    'a photo of a dog',
    'a photo of a frog',
    'a photo of a horse',
    'a photo of a ship',
    'a photo of a truck'
]

class CIFAR10WithPrompt(Dataset):
    """
    images -> [3, 32, 32], scaled to [-1, 1]
    labels -> int (0~9)
    prompts -> str
    """
    def __init__(self, root: str='./data', train: bool = True, transform=None, download: bool = True):
        self.dataset = datasets.CIFAR10(root=root, 
                                        train=train, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) if transform is None else transform,
                                        download=download)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        prompt = CIFAR10_CLASSES_PROMPTS[label]
        return img, label, prompt
    

def get_dataloader(batch_size: int = 64, 
                   shuffle: bool = True, 
                   num_workers: int = 2, 
                   root: str = './data', 
                   train: bool = True):
    dataset = CIFAR10WithPrompt(root=root, train=train)
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=shuffle, 
                        num_workers=num_workers,
                        pin_memory=True)
    return loader