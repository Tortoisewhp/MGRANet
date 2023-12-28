#
import torch
import torch.nn as nn
from toolbox.models.MGRANet.segformer.mix_transformer import mit_b0
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def get_upcog(uncertainty_map, num_points):
    R, _, H, W = uncertainty_map.shape
    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = point_indices % W
    point_coords[:, :, 1] = point_indices // W
    return point_indices, point_coords

def ps(input, point_indices):
    N, C, H, W = input.shape
    point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
    flatten_input = input.flatten(start_dim=2)
    sampled_feats = flatten_input.gather(dim=2, index=point_indices).view_as(point_indices)
    return sampled_feats

class GCN(nn.Module):
    def __init__(self, node_num, node_fea):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.node_fea = node_fea
        self.conv_adj = nn.Conv1d(self.node_num, self.node_num, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(self.node_num)
        self.conv_wg = nn.Conv1d(self.node_fea, self.node_fea, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(self.node_fea)
        self.relu = nn.ReLU()
    def forward(self, x):
        z = self.conv_adj(x)
        z = self.bn_adj(z)
        z = self.relu(z)
        z = z+x
        z = z.transpose(1, 2).contiguous()
        z = self.conv_wg(z)
        z = self.bn_wg(z)
        z = self.relu(z)
        z = z.transpose(1, 2).contiguous()
        return z
#ARE
class ARE(nn.Module):
    def __init__(self, inplance, num_points, thresholds=0.8):
        super(ARE, self).__init__()
        self.num_points = num_points
        self.thresholds = thresholds
        self.gcn = GCN(num_points, inplance)
    def forward(self, x, edge):
        B, C, H, W = x.size()
        edge[edge < self.thresholds] = 0
        edge_index, point_coords = get_upcog(edge, self.num_points)
        gcn_features = ps(x, edge_index).permute(0, 2, 1)
        gcn_features_reasoned = self.gcn(gcn_features)
        gcn_features_reasoned = gcn_features_reasoned.permute(0, 2, 1)
        edge_index = edge_index.unsqueeze(1).expand(-1, C, -1)
        final_features = x.reshape(B, C, H * W).scatter(2, edge_index, gcn_features_reasoned.float()).view(B, C, H, W)
        return final_features

class SGCN(nn.Module):
    def __init__(self, plane):
        super(SGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return nn.Sigmoid()(out)

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class DGCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(DGCN, self).__init__()
        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))
    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x
    def forward(self, feat):
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        out = self.final(torch.cat((spatial_local_feat, g_out), 1))
        return nn.Sigmoid()(out)

class MGRA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MGRA, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )

        self.dualgcn0 = DGCN(out_channel)
        self.dualgcn1 = DGCN(out_channel)
        self.dualgcn2 = DGCN(out_channel)
        self.dualgcn3 = DGCN(out_channel)
        self.dualgcn4 = DGCN(out_channel)

        self.conv_cat = BasicConv2d(5 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x0 = self.dualgcn0(x0)
        x1 = self.dualgcn1(x1)
        x2 = self.dualgcn2(x2)
        x3 = self.dualgcn3(x3)
        x4 = self.dualgcn4(x4)

        x1=x1*x0
        x2=x2*x1
        x3=x3*x2
        x4=x4*x3

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class decoder(nn.Module):
    def __init__(self, channel,highConv1_Channel,refineConv1_Channel):
        super(decoder, self).__init__()
        self.refineConv1 = nn.Conv2d(refineConv1_Channel, channel, kernel_size=1, bias=False)
        self.highConv1 = nn.Conv2d(highConv1_Channel, channel, kernel_size=1, bias=False)

        self.edge_out_pre = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.edge_out = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, bias=False))

        self.body_out_pre = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.body_out = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, bias=False))

        self.final_seg_out_pre = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.final_seg_out = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1, bias=False))

    def forward(self, xin, x5,x2):
        fine_size=(416 ,416)
        xin=self.refineConv1(xin)
        x5=self.highConv1(x5)
        xin = F.interpolate(xin, size=fine_size, mode='bilinear', align_corners=True)
        x5 = F.interpolate(x5, size=fine_size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=fine_size, mode='bilinear', align_corners=True)

        x=xin+x5+x2
        seg_final_outs=self.final_seg_out(self.final_seg_out_pre(x))
        seg_body_outs=self.body_out(self.body_out_pre(x))
        seg_edge_outs=self.edge_out(self.edge_out_pre(x))
        return nn.Sigmoid()(seg_final_outs), nn.Sigmoid()(seg_body_outs), nn.Sigmoid()(seg_edge_outs),x

class MGRANet_student(nn.Module):
    def __init__(self, channel=32):
        super(MGRANet_student, self).__init__()
        # Backbone
        self.rgb = mit_b0()#b0 32 64 160 256
        self.depth = mit_b0()
        # Decoder
        self.are=ARE(channel, 96, 0.8)
        self.con1_1=nn.Conv2d(32,1,1,bias=False)
        self.mgra = MGRA(256,256)
        self.DL4 = decoder(channel, 256, 256)
        self.DL3 = decoder(channel, 160, channel)
        self.DL2 = decoder(channel, 64, channel)

    def forward(self, x, x_depth):
        x = self.rgb.forward_features(x)
        x_depth = self.depth.forward_features(x_depth)
        x1 = x[0]
        x1_depth = x_depth[0]
        x2 = x[1]
        x2_depth = x_depth[1]
        x3_1 = x[2]
        x3_1_depth = x_depth[2]
        x4_1 = x[3]
        x4_1_depth = x_depth[3]

        x1_1 = x1+x1_depth
        x2_1 = x2+x2_depth
        x3_1 = x3_1+x3_1_depth
        x4_1 = x4_1+x4_1_depth

        #fusedFeature=[x1_1,x2_1,x3_1,x4_1]
        #ARE
        edge=self.con1_1(x1_1)
        x1_1=self.are(x1_1.float(),edge)

        #MGRAM
        x4_2 = self.mgra(x4_1)

        #Decoder Layer (DL)
        seg_final_outs4, seg_body_outs4, seg_edge_outs4, x4 = self.DL4(x4_2, x4_1, x1_1)
        seg_final_outs3, seg_body_outs3, seg_edge_outs3, x3 = self.DL3(x4, x3_1, x1_1)
        seg_final_outs2, seg_body_outs2, seg_edge_outs2, x2 = self.DL2(x3, x2_1, x1_1)
        seg_final_outs, seg_body_outs, seg_edge_outs = [seg_final_outs4, seg_final_outs3, seg_final_outs2], [seg_body_outs4, seg_body_outs3,seg_body_outs2], [seg_edge_outs4, seg_edge_outs3, seg_edge_outs2]

        #return fusedFeature,[seg_final_outs, seg_body_outs, seg_edge_outs]
        return [seg_final_outs, seg_body_outs, seg_edge_outs]


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416).cuda()
    depth = torch.randn(1, 3, 416, 416).cuda()
    model = MGRANet_student().to(torch.device("cuda:0"))
    out = model(img, depth)
    for i in range(len(out[0])):
        print(out[0][i].shape)

