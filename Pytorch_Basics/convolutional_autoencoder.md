---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/PamelaVQ/Base-ML/blob/master/Pytorch_Basics/convolutional_autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="49y7DnS21Bpu" colab_type="text" -->
Create a simple autoencoder using pytorch
<!-- #endregion -->

```python id="sdB939JC1Kes" colab_type="code" colab={}
import torch
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.utils import make_grid
from torch import nn
from torchvision import transforms
from torch.utils.data import dataloader
from torch.autograd import Variable
```

```python id="HOFa_2_95HLk" colab_type="code" colab={}
transforms_list = [transforms.functional.adjust_brightness(img, brightness_factor=2),transforms.functional.hflip(img)]
dataset = MNIST(root='.', download=True, transform=transforms.Compose([transforms.ToTensor(), 
                                                                              transforms.Normalize([0.5], [0.5]), 
                                                                       ]))
```

```python id="WrbP1frp5_g2" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="fac4f543-9b46-4d3a-d694-17b875e32a6b"
# create a dataloader
batch_size = 128
mnist_data = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(dataset.data.shape)

```

```python id="z1zDxvFr7NA0" colab_type="code" colab={}
# create a convolutional autoencoder
class Encoder(nn.Module):
  def __init__(self, c, embedding_size): # [b, 1, 28, 28]
    super(Encoder, self).__init__()
    self.conv2d1 = nn.Conv2d(c, 10, 5, stride=2, padding=1)
    self.conv2d2 = nn.Conv2d(10, 20, 3, stride=2, padding=1)
    # self.maxpool = nn.MaxPool2d(3)
    self.conv2d3 = nn.Conv2d(20, 40, 5, stride=3, padding=1)
    self.fully = nn.Linear(160, embedding_size) 

  def forward(self, x):
    x = torch.relu(self.conv2d1(x))
    x = torch.relu(self.conv2d2(x))
    # x = self.maxpool(x)
    x = torch.relu(self.conv2d3(x))
    x = x.view(x.data.shape[0], 160)
    x = self.fully(x)
    return x

class Decoder(nn.Module):
  def __init__(self, c, input_size):
    super(Decoder, self).__init__()
    self.fully = nn.Linear(input_size, 160) 
    self.conv2d1 = nn.ConvTranspose2d(40, 20, 5, stride=4, padding=1)
    self.conv2d2 = nn.ConvTranspose2d(20, 10, 3, stride=2, padding=1)
    self.conv2d3 = nn.ConvTranspose2d(10, c, 6, stride=2, padding=1)

  def forward(self, x):
    x = self.fully(x)
    x = x.view(x.data.shape[0], 40, 2, 2)
    x = torch.relu(self.conv2d1(x))
    x = torch.relu(self.conv2d2(x))
    x = torch.relu(self.conv2d3(x))
    return x

```

```python id="GFddNTmqDUt8" colab_type="code" colab={}
channels = 1
encoder = Encoder(channels, embedding_size=40)
decoder = Decoder(channels, input_size=40)

autoencoder = nn.Sequential(encoder, decoder)
```

```python id="TU2_w43aNMgd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 68} outputId="a4092d7a-ed7d-4404-9a11-06a42e6e9064"
x = Variable(torch.ones(batch_size, channels, 28, 28))
e = encoder(x)
d = decoder(e)

print('Input\t ', list(x.data.shape))
print('Embedding', list(e.data.shape))
print('Output\t ', list(d.data.shape))
```

```python id="VhglA7RIDtph" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 255} outputId="c6964131-6630-4bc4-8c46-f76bec04715b"
model = autoencoder.cuda()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters())
model.train()
```

```python id="ORyGsuCW3jDP" colab_type="code" colab={}
# Create transforms
def img_transform(img):
  transforms_list = [transforms.functional.adjust_brightness(img, brightness_factor=2)]
  # transforms.RandomApply(transforms_list, p=0.5)(img)
  img =  transforms.Compose([transforms.ToPILImage(), transforms.RandomApply(transforms_list, p=0.5), 
                             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(img)
  # print(type(img))
  return img

def img_normalize(img):
  img =  transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  return img
```

```python id="3L_jcTtME5Kd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="f616f946-4b18-4181-c287-5f72f495fb6b"
# print(mnist_data.dataset.data[0].shape)
num_epochs = 100
epoch_loss = []
for epoch in range(num_epochs):
  batch_loss = []
  for batch_num, (data, _) in enumerate(mnist_data): # data = [128, 1, 28, 28]
    # print(data.shape)
    optim.zero_grad()
    # data = data.view(data.shape[0], -1) # data = [128, 784]
    # print(data.shape)
    # input = img_transform(data)
    # data = img_normalize(data)
    data = Variable(data).cuda()
    # print(data.shape)
    
    # print(input.shape)
    output = model(data)
    loss = loss_fn(output, data)
    loss.backward()
    optim.step()
    batch_loss.append(loss)
  epoch_loss.append(sum(batch_loss)/len(batch_loss))
  print(f'Epoch {epoch}:\tloss {epoch_loss[-1]:.4f}')
```

```python id="JQoqKeu0Z8OR" colab_type="code" colab={}
def to_img(x):
    x = 0.5 * x + 0.5
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x
```

```python id="WyNFcBZocfD1" colab_type="code" colab={}
import matplotlib.pyplot as plt
import numpy as np
def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
```

```python id="NUN0Cg5YGHF9" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 261} outputId="068c7394-d162-4e19-81ce-e8472fed4b61"
# print output
print_data = dataloader.DataLoader(dataset, batch_size=20, shuffle=True)
from PIL import Image
for batch_num, (data, _) in enumerate(print_data):
    img1 = to_img(data)
    data = Variable(img1).cuda()
    output = model(data)
    output = output.cpu()
    img = to_img(output)
    show(make_grid(torch.cat([img1, img])))
    break
```

```python id="8PReiJ7ydxQf" colab_type="code" colab={}

```
