from Net.modules import *
from Net.functional import batch_cov


class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=3):
        super(AdaptiveAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        b, c, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = F.relu(self.fc1(out))
        attn_scores = torch.sigmoid(self.fc2(out))
        return attn_scores


class XXXPNet_Basic(nn.Module):
    def __init__(self, channel_num=18):
        super(XXXPNet_Basic, self).__init__()

        self.attention_weight = None
        self.multi_head = None

        self.channel_in = channel_num
        self.dim_out = [16, 12]  # gai
        self.multi_num = [1, 4]
        self.kernel_size = 2
        # classes = 2 #gai
        classes = 5
        self.tcn_channels = 32
        self.tcn_width = 9
        self.attn_in = self.tcn_channels

        self.ReEig = ReEig()
        self.LogEig = LogEig()

        self.BiMap_Block1 = nn.Sequential(
            BiMap(self.channel_in, self.dim_out[0], self.dim_out[0], self.multi_num[0]),
            ReEig(),
            BiMap(self.channel_in, self.dim_out[0], self.dim_out[0], self.multi_num[0]),
            ReEig(),
            BatchNormSPD(momentum=0.1, n=self.dim_out[0]),
        )
        self.BiMap_Block2 = BiMap(
            self.channel_in, self.dim_out[0], self.dim_out[1], self.multi_num[1]
        )

        self.Temporal_Block = nn.Conv2d(
            1,
            self.tcn_channels,
            (self.kernel_size, self.multi_num[1] * self.dim_out[1] ** 2),
            stride=(1, self.dim_out[1] ** 2),
            padding=0,
        )

        # self.Attention = nn.Linear(self.tcn_width, self.tcn_width, bias=False)

        self.Attention = AdaptiveAttention(self.tcn_width)

        self.Classifier = nn.Linear(self.tcn_channels * self.tcn_width, classes)

        # self.Classifier = nn.Sequential(nn.Linear(self.tcn_channels * self.tcn_width, self.tcn_channels * self.tcn_width),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.tcn_channels * self.tcn_width, classes))

        self.MultiHeadAttention = MultiHeadAttention(32, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        batch_size, band_num, window_num, channel_num, dim = x.shape
        x = x.reshape(batch_size * window_num * band_num, channel_num, dim)
        x = x.permute(0, 2, 1)
        x = batch_cov(x)
        x = x.reshape(batch_size, window_num * band_num, channel_num, channel_num)

        x_log = self.BiMap_Block1(x)
        x_log, L = self.BiMap_Block2(x_log)

        x_log = self.LogEig(x_log)
        x_log = x_log.to(torch.float32)
        self.multi_head = x_log

        x_vec = x_log.view(batch_size * band_num, 1, window_num, -1)
        # print("x_vec size:", x_vec.size())
        x_vec = self.Temporal_Block(x_vec)
        x_vec = x_vec.view(batch_size, band_num, -1)
        # print("x_vec size:", x_vec.size())

        attention_weight = self.Attention(x_vec)

        self.attention_weight = attention_weight
        # print("self.attention_weight", self.attention_weight.shape)
        attn_output = x_vec * attention_weight.unsqueeze(-1)
        # print("attn_output", attn_output.size())
        attn_output = attn_output.view(batch_size, -1)
        # print("attn_output", attn_output.size())
        y = self.Classifier(attn_output)

        return y, L
