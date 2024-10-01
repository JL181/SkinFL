import tenseal as ts
import numpy as np
from prognosis.enc_network import CollapsedNets
from prognosis.enc_network import EncNet

def enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride):
    # 初始化用于监视测试损失和准确率的列表
    test_loss = 0.0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))

    n = 0
    for data, target in test_loader:
        dat = F.pad(data, (1, 1, 1, 1))
        x_enc, windows_nb = ts.im2col_encoding(
            context, dat.view(30, 30).tolist(), kernel_shape[0], kernel_shape[1], stride)
        enc_output = enc_model(x_enc, windows_nb)
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        loss = criterion(output, target)
        test_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1
    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')

    for label in range(2):
        print(
            f'Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% '
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )


model2 = CollapsedNets()
model2.load_state_dict(torch.load('./net.pth'), strict=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
kernel_shape = model2.conv1.kernel_size
stride = model2.conv1.stride[0]

# controls precision of the fractional part
bits_scale = 40

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=32768,
    coeff_mod_bit_sizes=[60, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 60]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

# Encrypted Evaluation
model2 = torch.load('./EncNet.pth')
model2.eval()

t3 = time()
enc_model = EncNet(model2)
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
t4 = time()
print("time=", t4 - t3)
