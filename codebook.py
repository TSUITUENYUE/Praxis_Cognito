import torch
import torch.nn.functional as F

class Codebook:
    def __init__(self,delta_init,update_rule):
        self.delta_init = delta_init
        self.len = 0
        self.intent = []
        self.delta = []
        self.num = []
        self.update_rule = update_rule

    def find(self, z):
        for i in self.intent:
            c_i = self.intent[i]
            delta_i = self.delta[i]
            if F.mse_loss(z, c_i) < delta_i:
                return i
        return -1

    def update(self, z):
        if self.update_rule == 'sphere':
            i = self.find(z)
            if i > 0:
                print("Already Existing Intention")
                c_i = self.intent[i]
                delta_i = self.delta[i]
                n_i = self.num[i]

                self.intent[i] = (c_i * n_i + z) / (n_i + 1)
                self.num[i] +=1
            else:
                print("No Existing Intention, Add New One")
                self.intent.append(z)
                self.delta.append(self.delta_init)
                self.num.append(1)

