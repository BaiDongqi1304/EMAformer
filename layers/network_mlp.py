import torch
from torch import nn

class NetworkMLP(nn.Module):
    def __init__(self, configs, d_model = 128):
        super(NetworkMLP, self).__init__()
        
        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = d_model
        # Patching
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        if self.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        # Linear Stream
        # MLP
        self.fc1 = nn.Linear(self.seq_len, self.pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(self.pred_len * 2)

        self.fc2 = nn.Linear(self.pred_len * 2, self.pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(self.pred_len // 2)

        self.fc3 = nn.Linear(self.pred_len // 2, self.d_model)

    def forward(self, x):
        # x: [Batch, Channel, Input]

        # Channel split
        B = x.shape[0] # Batch size
        C = x.shape[1] # Channel size
        I = x.shape[2] # Patch_num
        x = torch.reshape(x, (B*C, I)) # [Batch and Channel, Patch_num, Patch_len]

        # Linear Stream
        # MLP
        x = self.fc1(x)
        x = self.avgpool1(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.avgpool2(x)
        x = self.ln2(x)

        x = self.fc3(x) # [Batch * Channel, d_model]
        x = x.unsqueeze(1) #[Batch * Channel, 1, d_model]
        x = x.repeat(1, self.patch_num, 1) #[Batch * Channel, patch_num, d_model]

        # Channel concatination
        #x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        #x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x