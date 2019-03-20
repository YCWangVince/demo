import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as dataf
from torch.utils.data import Dataset
from PIL import Image
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime



dataset_num = 60000
lr_init_g = 0.0001
lr_init_d = 0.0001
Noise_dim = 100
batch_size = 250
Noise_num = 30000
batch_num = dataset_num/batch_size
train_path = './Dataset/train/'
real_label = 1
fake_label = 0
all_path =[]


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def load_data(data_path):
    signal = os.listdir(data_path)
    for fsingal in signal:
        filepath = data_path + fsingal
        all_path.append(filepath)

    sample_path = random.sample(all_path, dataset_num)
    random.shuffle(sample_path)
    count = len(sample_path)
    data_x = np.empty((count,1,28,28),dtype='float32')
    data_y = []


    i = 0

    for item in sample_path:
        img = cv2.imread(item,0)
        img = cv2.resize(img,(28,28))
        arr = np.asarray(img,dtype='float32')
        data_x[i,:,:,:] = arr
        i += 1
        data_y.append(real_label)

    data_x = data_x/255.0
    data_y = np.asarray(data_y)
    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)

    train_data = dataf.TensorDataset(data_x,data_y)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

    return train_loader

# train_data = dset.ImageFolder(root='./Dataset/train/',
#                               transform= transforms.Compose([
#                                   transforms.Resize(28),
#                                   transforms.CenterCrop(28),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize(0.1291364, 0.28719229)
#                               ]))
#
# train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True, workers=workers)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# real_batch = next(iter(train_loader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
        Flatten(),
        nn.Linear(Noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7*7*128),
        nn.BatchNorm1d(7*7*128),
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Tanh()
            )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1,32, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32,64, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(4*4*64, 4*64),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(4*64, 1),
        nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


train_loader = load_data(train_path)

Gnet = G().to(device)
Gnet.initialize_weights()

print(Gnet)

Dnet = D().to(device)
Dnet.initialize_weights()

print(Dnet)

optimizer_G = optim.Adam(Gnet.parameters(), lr=lr_init_g, betas =(0.9, 0.999), eps= 10e-8)
optimizer_D = optim.Adam(Dnet.parameters(), lr=lr_init_d, betas =(0.9, 0.999), eps= 10e-8)


criterion = nn.BCELoss()
max_epochs = 100


def fixed_noise():
    return torch.randn(dataset_num, Noise_dim, 1, 1, device=device)


print("Starting Training...")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
log_dir = os.path.join('./Result/', time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)
noise = np.random.randn(dataset_num, Noise_dim, 1, 1)
noise_test_1 = np.random.randn(batch_size, Noise_dim, 1, 1)
noise_test_1 = noise_test_1/noise_test_1.max()
noise_test_1 = torch.Tensor(noise_test_1).to(device)
test = fixed_noise()

writer.add_text('Normalized input\n'+'parameters:', 'dataset_num = ' + str(dataset_num) + '\nlr_init_g = ' + str(lr_init_g) +
                '\nlr_init_d = ' + str(lr_init_d) + '\nNoise_dim = ' + str(Noise_dim) +
'\n Batch_size = ' + str(batch_size))

for epoch in range(max_epochs):
    for i, data in enumerate(train_loader,0):
        noise_i = noise[int(i*batch_size):int((i+1)*batch_size),:,:,:]
        noise_i = noise_i/noise_i.max()
        noise_i = torch.Tensor(noise_i).to(device)
        optimizer_D.zero_grad()
        inputs = data[0].to(device)
        output = Dnet(inputs).view(-1)
        b_size = output.size(0)
        label = torch.full((b_size,), real_label, device=device)
        errorD_real = criterion(output, label)
        errorD_real.backward()
        D_x = output.mean().item()
        # optimizer_G.zero_grad()
        fake = Gnet(noise_i)
        label.fill_(fake_label)
        output = Dnet(fake.detach()).view(-1)
        errorD_fake = criterion(output, label)
        errorD_fake.backward()
        D_G_z1 =output.mean().item()
        errD = errorD_real + errorD_fake
        optimizer_D.step()
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(x): %.4f\tD(G(z)): %.4f '
                  % (epoch, max_epochs, i, len(train_loader),
                     errD,  D_x, D_G_z1))
    noise_index = np.random.randint(noise.shape[0], size=Noise_num)
    noise_s = noise[noise_index, :, :, :]
    for i in range(int(Noise_num/batch_size)):
        optimizer_G.zero_grad()
        noise_i = noise_s[int(i * batch_size):int((i + 1) * batch_size), :, :, :]
        noise_i = noise_i / noise_i.max()
        noise_i = torch.Tensor(noise_i).to(device)
        fake = Gnet(noise_i)
        label.fill_(real_label)
        output = Dnet(fake).view(-1)
        errorG = criterion(output, label)
        errorG.backward()
        D_G_z2=output.mean().item()
        optimizer_G.step()
        if i%10==0:
            print('[%d/%d][%d/%d]\tErrorG:%.4f\tD(G(z)): %.4f ' % (epoch, max_epochs, i, Noise_num/batch_size, errorG, D_G_z2))
    with torch.no_grad():
        fake = Gnet(noise_test_1)
        fake = vutils.make_grid(fake, normalize=True, scale_each=True)
        if epoch == 0 or epoch == 9 or epoch == 19 or epoch == 49 or epoch == 79 or epoch == 99 or epoch == 119 or epoch == 149:
            writer.add_image('Image', fake, epoch)
writer.close()
