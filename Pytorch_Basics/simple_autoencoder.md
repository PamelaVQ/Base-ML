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
<a href="https://colab.research.google.com/github/PamelaVQ/Base-ML/blob/master/Pytorch_Basics/simple_autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="49y7DnS21Bpu" colab_type="text" -->
Create a simple autoencoder using pytorch
<!-- #endregion -->

```python id="sdB939JC1Kes" colab_type="code" colab={}
import torch
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch import nn
from torchvision import transforms
from torch.utils.data import dataloader
from torch.autograd import Variable
```

```python id="HOFa_2_95HLk" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 170} outputId="fc962dc5-ff43-49de-fa82-ce792bc09048"
dataset = MNIST(root='.', download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
dataset
```

```python id="WrbP1frp5_g2" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="fec6b507-931d-4812-b6eb-b9b55629cb0d"
# create a dataloader
batch_size = 128
mnist_data = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(dataset.data.shape, dataset.data.shape[1], dataset.data.shape[2])
```

```python id="z1zDxvFr7NA0" colab_type="code" colab={}
# create a simple autoencoder
class Encoder(nn.Module):
  def __init__(self, input_1, input_2):
    super(Encoder, self).__init__()
    self.linear1 = nn.Linear(input_1 * input_2, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 12)
    self.linear4 = nn.Linear(12, 3)

  def forward(self, x):
    x = torch.relu(self.linear1(x))
    x = torch.relu(self.linear2(x))
    x = torch.relu(self.linear3(x))
    x = torch.relu(self.linear4(x))
    # x = nn.Tanh()
    return x

class Decoder(nn.Module):
  def __init__(self, output_1, output_2):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(3, 12)
    self.linear2 = nn.Linear(12, 64)
    self.linear3 = nn.Linear(64, 128)
    self.linear4 = nn.Linear(128, output_1 * output_2)   

  def forward(self, x):
    x = torch.relu(self.linear1(x))
    x = torch.relu(self.linear2(x))
    x = torch.relu(self.linear3(x))
    x = torch.relu(self.linear4(x))
    # x = nn.Tanh(x)
    return x

```

```python id="GFddNTmqDUt8" colab_type="code" colab={}
encoder = Encoder(dataset.data.shape[1], dataset.data.shape[2])
decoder = Decoder(dataset.data.shape[1], dataset.data.shape[2])

autoencoder = nn.Sequential(encoder, decoder)
```

```python id="TU2_w43aNMgd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 68} outputId="57db457b-f02c-4714-d1bb-26a48f1c50c0"
x = Variable(torch.ones(batch_size, 784))
e = encoder(x)
d = decoder(e)

print('Input\t ', list(x.data.shape))
print('Embedding', list(e.data.shape))
print('Output\t ', list(d.data.shape))
```

```python id="VhglA7RIDtph" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 255} outputId="40c9e386-8cb3-4179-b6e8-686bd3f22d6d"
model = autoencoder.cuda()
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters())
model.train()
```

```python id="3L_jcTtME5Kd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="6be77ef4-6fd0-4edd-c1bb-efd6e2629ff8"
num_epochs = 100
epoch_loss = []
for epoch in range(num_epochs):
  batch_loss = []
  for batch_num, (data, _) in enumerate(mnist_data): # data = [128, 1, 28, 28]
    optim.zero_grad()
    data = data.view(data.shape[0], -1) # data = [128, 784]
    data = Variable(data).cuda()
    output = autoencoder(data)
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

```python id="NUN0Cg5YGHF9" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 261} outputId="19a80ed6-4656-4462-8340-f21c53ae611a"
# print output
# also test with cuda
print_data = dataloader.DataLoader(dataset, batch_size=20, shuffle=True)
model = autoencoder.cuda()
from PIL import Image
for batch_num, (data, _) in enumerate(print_data):
    img1 = to_img(data)
    # show(make_grid(img))
    data = data.view(data.shape[0], -1)
    data = Variable(data).cuda()
    output = autoencoder(data)
    output = output.cpu()
    img = to_img(output)
    show(make_grid(torch.cat([img1, img])))
    break
```

```python id="8PReiJ7ydxQf" colab_type="code" colab={}

```
