import tenseal as ts
import numpy as np

class CKKSCipher:
    def __init__(self):
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=1024, coeff_mod_bit_sizes=[60, 40, 60])
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()

    def encrypt(self, value):
        return ts.ckks_vector(self.context, value).serialize()

    def decrypt(self, value):
        return np.array(ts.CKKSVector.load(self.context, value).decrypt())

    def enc_batch(self, value):
        batch_size = 128
        batch_num = int(np.ceil(len(value) / batch_size))
        cipher_list = []
        ser_cipher_size = 0
        # print("len value=", len(value), 'size value=', convert_size(batch_size), "batchN=", batch_num)
        for i in range(batch_num):
            cipher = ts.ckks_vector(self.context, value[i * batch_size: (i + 1) * batch_size])
            ser_cipher = cipher.serialize()
            cipher_list.append(ser_cipher)
            ser_cipher_size = len(ser_cipher)
        # print("len_ser_cipher=", ser_cipher_size, 'size cipher=',convert_size(ser_cipher_size))
        return cipher_list, batch_num, ser_cipher_size

    def sum_batch(self, arr, idx_weights):
        batch_num = len(arr[0])
        res_list = []
        for batch in range(batch_num):
            res = ts.CKKSVector(self.context, [0])  # 初始化一个加密的零向量
            for client_cipher, weight in zip(arr, idx_weights):
                loaded_vector = ts.CKKSVector.load(self.context, client_cipher[batch])
                res += loaded_vector * weight
            res_list.append(res.serialize())  # 确保使用serialize
        return res_list

    def dec_batch(self, serial_list):
        plain_list = []

        for cipher_serial in serial_list:
            plain = ts.CKKSVector.load(self.context, cipher_serial).decrypt()
            plain_list.extend(plain)
        return np.array(plain_list)