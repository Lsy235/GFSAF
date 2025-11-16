import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import time


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('   总参数数量: %.2fM' % (total / 1e6))


class TestModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(TestModel, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hid = F.relu(self.linear1(x))
        hid = F.relu(self.linear2(hid))
        hid = F.relu(self.linear3(hid))
        output = F.sigmoid(self.linear4(hid))
        return output


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), 'checkpoint/' + name)
        return name

    def forward(self, *input):
        pass


class ImgModule(BasicModule):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024 * 8, mid_num2=1024 * 8, hiden_layer=3):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        # self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out

if __name__ == "__main__":
    # model1 = TestModel(32*32, 10)
    # print_model_parm_nums(model1)
    model2 = ImgModule(25, 25)
    print_model_parm_nums(model2)
