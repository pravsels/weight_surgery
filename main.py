
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision.models as models
import torch

BATCH_SIZE = 64

def stack(tensor, times=3):
    return (torch.cat([tensor]*times, dim=0))

transformations = transforms.Compose([transforms.Resize((221, 221)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485], std=[0.229]),
                                      stack
                                    ])

mnist_data = MNIST(root=r'./MNIST',
                   download=False,
                   transform=transformations)

data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

batch_images, batch_labels = next(iter(data_loader))
embeds = vgg16(batch_images)

print('Input images: ' + str(batch_images.shape))
print('Embeddings : ' + str(embeds.shape))
