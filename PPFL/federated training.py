from network.network import net
from Cryptographic.PFHE import CKKSCipher
from PPFL.enc_Aggregation import enc_federated_averaging
from PPFL.enc_Aggregation import enc_fedNova_aggregation
from PPFL.enc_Aggregation import enc_fedProx_aggregation
from PPFL.enc_Aggregation import enc_scaffold_aggregation

def train_local(model, train_loader, criterion, optimizer, n_epochs):
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        correct_new = 0
        total_new = 0
        t1 = time()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_new += target.size(0)
            correct_new += (predicted == target).sum().item()
        t2 = time()
        train_loss = train_loss / len(train_loader)
        acc_new = 100 * correct_new / total_new

        print('Epoch: {} \tTraining Loss: {:.6f} \tCorrect: {} \tAccuracy: {:.3f} \ttime: {:.3f}'.format(epoch,
                                                                                                         train_loss,
                                                                                                         correct_new,
                                                                                                         acc_new,
                                                                                                         t2 - t1))
    print('Local Finished Training')
    return model

global_model = net()
criterion = torch.nn.CrossEntropyLoss()
global_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.0001)

cipher = CKKSCipher()

global_epochs = g_epochs
local_epochs = l_epochs
for global_epoch in range(global_epochs):
    party_n = 1
    local_models = []
    for local_loader in local_loaders:
        print(f'\nGlobal Epoch {global_epoch + 1}/{global_epochs} Party {party_n}/{5}')
        party_n += 1
        local_model = deepcopy(global_model)
        local_optimizer = torch.optim.Adam(local_model.parameters(), lr=0.0001)
        local_model = train_local(local_model, local_loader, criterion, local_optimizer, n_epochs=local_epochs)
        local_models.append(local_model)
    global_model, total_cipher_num, ser_cipher_size = enc_federated_averaging(global_model, local_models, cipher)
    test_accuracy = test(global_model, test_loader, criterion)
