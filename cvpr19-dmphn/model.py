import torch.nn as nn
from decoder import Decoder
from encoder import Encoder

class DMPHNModel(nn.Module):
    def __init__(self, level=4, device='cuda'):
        super(DMPHNModel, self).__init__()
        self.encoders = []
        self.decoders = []
        self.level = level
        for i in range(level):
            self.encoders.append(Encoder().to(device))
            self.decoders.append(Decoder().to(device))
        self.params = nn.ParameterList([self.encoders[i].parameters() for i in range(level)]+[self.decoders[i].parameters() for i in range(level)])

    def forward(self, x):
        # x structure (B, h, w, c)
        # from bottom to top
        tmp_out = []
        tmp_feature = []
        for i in range(self.level):
            currentlevel = self.level - i - 1  # 3,2,1,0
            # For level 4(i.e. i = 3), we need to divide the picture into 2^i parts without any overlaps
            num_parts = 2 ** currentlevel
            rs = []
            if currentlevel == 3:
                rs = self.divide(x, 2, 4)
                for j in range(num_parts):
                    tmp_feature.append(self.Encoder[currentlevel](rs[j]))  # each feature is [B, H, W, C=128]
                # combine the output
                tmp_feature = self.combine(tmp_feature, comb_dim=2)
                for j in range(num_parts/2):
                    tmp_out.append(self.Decoder[currentlevel](tmp_feature[j]))
            elif currentlevel == 2:
                rs = self.divide(x, 2, 2)
                for j in len(rs):
                    rs[j] = rs[j] + tmp_out[j]
                    tmp_feature.append(self.Encoder[currentlevel](rs[j]))
                tmp_feature = self.combine(tmp_feature, comb_dim=1)
                for j in range(num_parts/2):
                    tmp_out.append(self.Decoder[currentlevel](tmp_feature[j]))
            elif currentlevel == 1:
                rs = self.divide(x, 1, 2)
                for j in len(rs):
                    rs[j] = rs[j] + tmp_out[j]
                    tmp_feature.append(self.Encoder[currentlevel](rs[j]))
                tmp_feature = self.combine(tmp_feature, comb_dim=2)
                for j in range(num_parts/2):
                    tmp_out.append(self.Decoder[currentlevel](tmp_feature[j]))
            else:
                x += tmp_feature[0]
                x = self.Decoder[currentlevel](self.Encoder[currentlevel](x))
        return x
    
    def combine(self, x, comb_dim=1):
        """[将数组逐两个元素进行合并并且返回]

        Args:
            x ([tensor array]): [输出的tensor数组]
            comb_dim (int, optional): [合并的维度，从高度合并则是1，宽度合并则是2]. Defaults to 1.

        Returns:
            [tensor array]: [合并后的数组，长度变为一半]
        """        
        rs = []
        for i in range(int(len(x)/2)):
            rs.append(torch.cat((x[2*i], x[2*i+1]),dim=comb_dim))
        return rs

    def divide(self, x, h_parts_num, w_parts_num):
        """ 该函数将BxHxWxC的输入进行切分, 本质上是对每一张图片进行分块
            这里直接针对多维数组进行操作

        Args:
            x (Torch Tensor): input torch tensor (e.g. [Batchsize, Heights, Width, Channels])
            h_parts_num (int): The number of divided parts on heights
            w_parts_num (int): The number of divided parts on width

        Returns:
            [A list]: h_parts_num x w_parts_num 's tensor list, each one has [B, H/h_parts_num, W/w_parts_num, Channels] structure
        """                
        rs = []
        for i in range(h_parts_num):
            tmp = x.chunk(h_parts_num, dim=1)[i]
            for j in range(w_parts_num):
                rs.append(tmp.chunk(w_parts_num,dim=2)[j])
        return rs