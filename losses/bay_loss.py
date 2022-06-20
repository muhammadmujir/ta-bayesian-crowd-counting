from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        print("Prob List Shape: {}:{}:{}".format(len(prob_list), len(prob_list[0]), len(prob_list[0][0])))
        print("Target List Shape: {}:{}".format(len(target_list), len(target_list[0])))
        print("Prediction Density Shape: ", pre_density.shape)
        print("target list: ", target_list)
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                # len (prob) --> sama dengan banyak orang
                N = len(prob)
                print("LEN PROB", N)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    # assign nilai 0 ke indeks terakhir
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                print("PROB: ", prob.shape)
                print("PRE_DENSITY: ", pre_density[idx].view((1, -1)).shape)
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector
                print("PRE_COUNT SHAPE: ", pre_count.shape)
            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss



