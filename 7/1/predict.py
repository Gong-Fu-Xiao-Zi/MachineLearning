import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio

class AdaptiveNet(torch.nn.Module):
        def __init__(self, input_height, input_width, conv1_channels, conv2_channels, dense_units):
            super(AdaptiveNet, self).__init__()
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, conv1_channels, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2))
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(conv1_channels, conv2_channels, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(conv2_channels, conv2_channels, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            self.height = input_height
            self.width  = input_width

            # Calculate output size to define the input for the fully connected layer
            conv_output_size = self.compute_conv_output_size(input_height, input_width)

            # Define the fully connected layer using the calculated output size
            self.dense = torch.nn.Sequential(
                torch.nn.Linear(conv2_channels * conv_output_size[0] * conv_output_size[1], dense_units),
                torch.nn.ReLU(),
                torch.nn.Linear(dense_units, 10)
            )

        # Function to compute output size based on input dimensions
        def compute_conv_output_size(self, height, width):
            """ Helper function to compute size after convolutions and pooling """
            def conv2d_output_size(size, kernel_size=3, stride=1, padding=1):
                return (size - kernel_size + 2 * padding) // stride + 1

            # conv1 + pool: height, width = input_size // 2
            height = conv2d_output_size(height) // 2
            width = conv2d_output_size(width) // 2

            # conv2 + pool
            height = conv2d_output_size(height) // 2
            width = conv2d_output_size(width) // 2

            # conv3 + pool
            height = conv2d_output_size(height) // 2
            width = conv2d_output_size(width) // 2

            return height, width

        def forward(self, x):
            conv1_out = self.conv1(x)
            conv2_out = self.conv2(conv1_out)
            conv3_out = self.conv3(conv2_out)
            res = conv3_out.view(conv3_out.size(0), -1)
            out = self.dense(res)
            return out

## 加载.mat文件，创建训练和测试网络的数据集
mnist_train = sio.loadmat('../data/mnist_train.mat')
train_images = mnist_train['images']
train_labels = mnist_train['labels']
mnist_test = sio.loadmat('../data/mnist_test.mat')
test_images = mnist_test['images']
test_labels = mnist_test['labels']
# 转换为 PyTorch 张量
train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).squeeze()  # 压缩标签维度

test_images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).squeeze()  # 压缩标签维度

# 创建 TensorDataset
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

## 训练网络
model = AdaptiveNet(input_height=28,input_width=28,conv1_channels=32, conv2_channels=64, dense_units=128)

loss_func = torch.nn.CrossEntropyLoss()

state_dict = torch.load('training.pth')

model.load_state_dict(state_dict)

# evaluation--------------------------------
model.eval()
eval_loss = 0.
eval_acc = 0.

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_loader.dataset)), eval_acc / (len(test_loader.dataset))))