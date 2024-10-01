model.load_state_dict(torch.load('./net.pth'))
_ = torch.manual_seed (2025)
# shape of intermediate "input" image
in_channels = 20
h = 13
w = h

#convolution parameters
out_channels = 50
kernel = 5
stride = 1
padding = 0

#fc paramaters
in_features = 200
out_features = 100

#LAYERS DEFINITION
#conv2 = torch.nn.Conv2d (in_channels, out_channels, kernel, stride)
#fc1 = torch.nn.Linear (in_features, out_features)
pool1 = nn.AvgPool2d(3, stride=2, padding=1)
conv2 = model.conv2
fc1 = model.fc1

# create collapsed bias from conv2 and fc1
bias = fc1 (torch.flatten (pool1(conv2 (pool1(torch.zeros (1, in_channels, h, w))))))
bias = ((torch.sub(bias, batch.running_mean))/torch.sqrt(torch.add(batch.running_var, batch.eps))) * batch.weight + batch.bias

# create collapsed weight from conv2 and fc1 (and bias)
# batch of images, each with only a single pixel turned on
n_pixels = in_channels * h * w   # number of pixels (including channels) in input image
pixel_batch = torch.eye (n_pixels).reshape (n_pixels, in_channels, h, w)
weight = (batch(fc1 (torch.flatten (pool1(conv2 (pool1(pixel_batch))), 1))) - bias).T

# create collapsed Linear
fcnew = torch.nn.Linear (n_pixels, out_features)   # Linear of correct shape

# copy in collapsed weight and bias
with torch.no_grad():
  _ = fcnew.weight.copy_ (weight)
  _ = fcnew.bias.copy_ (bias)


class CollapsedNets(torch.nn.Module):
    def __init__(self, hidden=100, output=10):
        super(CollapsedNets, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2, 1)
        self.fcnew = nn.Linear(3380, hidden)
        with torch.no_grad():
            _ = self.fcnew.weight.copy_(weight)
            _ = self.fcnew.bias.copy_(bias)
        self.fc2 = nn.Linear(hidden, output)
        self.hidden_size = hidden


    def forward(self, x):
        x = self.conv1(x)
        x = 0.125 * x * x + 0.5 * x + 0.25
        x = torch.flatten(x, 1)
        x = self.fcnew(x)
        x = 0.125 * x * x + 0.5 * x + 0.25
        x = self.fc2(x)
        return x

class EncNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fcnew_weight = torch_nn.fcnew.weight.T.data.tolist()
        self.fcnew_bias = torch_nn.fcnew.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)

        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        enc_x = enc_x.square()
        enc_x = enc_x.mm(self.fcnew_weight) + self.fcnew_bias
        enc_x = enc_x.square()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)





