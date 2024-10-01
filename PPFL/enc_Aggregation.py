def enc_federated_averaging(global_model, local_models, cipher):
    global_dict = global_model.state_dict()
    total_cipher_num = 0  # 密文数量是所有batch_num的总和
    ser_cipher_size = 0  # 密文参数量，通过convert_size(ser_cipher_size)得到大小KB

    for key in global_dict.keys():
        param_list = []
        for model in local_models:
            param = model.state_dict()[key].flatten().numpy()
            encrypted_param, batch_num, ser_cipher_size = cipher.enc_batch(param)
            param_list.append(encrypted_param)
            total_cipher_num += batch_num  # 累加密文数量
        encrypted_avg = cipher.sum_batch(param_list, [1.0 / len(local_models)] * len(local_models))
        global_dict[key] = torch.tensor(cipher.dec_batch(encrypted_avg)).view(global_dict[key].shape)

    global_model.load_state_dict(global_dict)

    return global_model, total_cipher_num, ser_cipher_size


def enc_fedNova_aggregation(global_model, local_models, local_steps, cipher):
    global_dict = global_model.state_dict()
    total_cipher_num = 0  # 密文数量是所有batch_num的总和
    ser_cipher_size = 0  # 密文参数量，通过convert_size(ser_cipher_size)得到大小KB

    for key in global_dict.keys():
        param_list = []
        for i, model in enumerate(local_models):
            update = (model.state_dict()[key] - global_dict[key]).flatten().numpy() * local_steps[i]
            encrypted_param, batch_num, ser_cipher_size = cipher.enc_batch(update)
            param_list.append(encrypted_param)
            total_cipher_num += batch_num  # 累加密文数量

        # 获取加密的平均更新
        encrypted_avg = cipher.sum_batch(param_list, [1.0 / sum(local_steps)] * len(local_models))

        # 解密并计算全局更新
        decrypted_updates = [cipher.dec_batch(encrypted_avg)]

        # 更新全局模型参数
        global_dict[key] += torch.tensor(decrypted_updates).view(global_dict[key].shape)
    global_model.load_state_dict(global_dict)
    return global_model, total_cipher_num, ser_cipher_size


def enc_fedProx_aggregation(global_model, local_models, cipher):
    global_dict = global_model.state_dict()
    total_cipher_num = 0  # 密文数量是所有batch_num的总和
    ser_cipher_size = 0  # 密文参数量，通过convert_size(ser_cipher_size)得到大小KB

    for key in global_dict.keys():
        param_list = []
        for model in local_models:
            encrypted_param, batch_num, ser_cipher_size = cipher.enc_batch(model.state_dict()[key].flatten().numpy())
            param_list.append(encrypted_param)
            total_cipher_num += batch_num  # 累加密文数量
        encrypted_avg = cipher.sum_batch(param_list, [1.0 / len(local_models)] * len(local_models))
        global_dict[key] = torch.tensor(cipher.dec_batch(encrypted_avg)).view(global_dict[key].shape)

    global_model.load_state_dict(global_dict)
    return global_model, total_cipher_num, ser_cipher_size


def enc_scaffold_aggregation(global_model, local_models, global_control, local_controls, ckks_cipher):
    global_dict = global_model.state_dict()
    global_ctrl_dict = {k: torch.zeros_like(v) for k, v in global_dict.items()}
    num_clients = len(local_models)
    total_cipher_num = 0  # 密文数量是所有batch_num的总和
    ser_cipher_size = 0  # 密文参数量，通过convert_size(ser_cipher_size)得到大小KB

    # 加密本地模型更新和控制变量
    encrypted_updates = {key: [] for key in global_dict.keys()}
    encrypted_controls = {key: [] for key in global_dict.keys()}
    for key in global_dict.keys():
        for i in range(num_clients):
            update = (local_models[i].state_dict()[key] - global_dict[key]).numpy().flatten()
            control_diff = (0.000001 * (local_controls[i][key] - global_control[key])).numpy().flatten()
            combined_update = update + control_diff  # 将更新和控制变量差异合并

            # 对合并的更新进行加密
            encrypted_param, batch_num, ser_cipher_size = ckks_cipher.enc_batch(combined_update)
            encrypted_updates[key].append(encrypted_param)
            total_cipher_num += batch_num

            # 对控制变量的更新进行加密
            encrypted_ctrl, batch_num, ser_cipher_size = ckks_cipher.enc_batch(control_diff)
            encrypted_controls[key].append(encrypted_ctrl)
            # total_cipher_num += batch_num

    # 聚合加密的更新和控制变量
    for key in global_dict.keys():
        global_update_encrypted = ckks_cipher.sum_batch(encrypted_updates[key], [1.0 / num_clients] * num_clients)
        global_ctrl_encrypted = ckks_cipher.sum_batch(encrypted_controls[key], [1.0 / num_clients] * num_clients)

        global_update = ckks_cipher.dec_batch(global_update_encrypted)
        global_control_update = ckks_cipher.dec_batch(global_ctrl_encrypted)

        global_dict[key] += torch.tensor(global_update).reshape(global_dict[key].shape)
        global_ctrl_dict[key] += torch.tensor(global_control_update).reshape(global_ctrl_dict[key].shape)

    global_model.load_state_dict(global_dict)
    for k in global_control.keys():
        global_control[k].data.copy_(global_ctrl_dict[k])

    return global_model, global_control, total_cipher_num, ser_cipher_size