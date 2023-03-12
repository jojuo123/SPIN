import torch
import torch.nn as nn
import numpy as np
from utils.geometry import rot6d_to_rotmat
from body_measurements import BodyMeasurements

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model=1, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper # n_att x B x 1 x 1024

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # n_att x B x 1 x d_v
        return output, attn

class Regressor_LabelAttention(nn.Module):
    def __init__(self, d_qk=512, d_v=512, d_model=1, n_attribute=1, n_layer=1, attention_dropout=0.1, residual_dropout=0.1, **kwargs):
        super().__init__()

        self.feature_input_dim = 2205
        self.attribute_q_proj = nn.Linear(1, d_qk)
        self.attribute_k_proj = nn.Linear(1, d_qk)
        self.attribute_v_proj = nn.Linear(1, d_v)
        nn.init.xavier_uniform_(self.attribute_q_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.attribute_k_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.attribute_v_proj.weight, gain=0.01)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.proj = nn.Linear(d_v, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        self.residual_dropout = nn.Dropout(residual_dropout)

        self.n_attribute = n_attribute
        self.d_qk = d_qk
        self.d_v = d_v

    def split_qkv(self, inp, attribute):
        len_inp = inp.size(0)
        # print(inp.shape)
        v_inp_repeated = inp.repeat(self.n_attribute, 1, 1, 1).view(self.n_attribute, len_inp, -1, inp.size(-1))
        k_inp_repeated = v_inp_repeated
        # attribute: n_att x B x 1 x d_k
        #k_repeat: n_att x B x 1024 x 1
        # q_s = torch.bmm(k_inp_repeated, attribute['q'])
        # k_s = torch.bmm(k_inp_repeated, attribute['k'])
        # v_s = torch.bmm(k_inp_repeated, attribute['v'])
        # q_s = torch.einsum('nbcd,nbdk->nbck',k_inp_repeated, attribute['q']) # n_att x B x 1024 x d_k
        q_s = attribute['q'] # n_att x B x 1 x d_k
        k_s = torch.einsum('nbcd,nbdk->nbck',k_inp_repeated, attribute['k']) # n_att x B x 1024 x d_k
        v_s = torch.einsum('nbcd,nbdk->nbck',k_inp_repeated, attribute['v'])
        return q_s, k_s, v_s
    
    def combine_v(self, outputs):
        n_attribute = self.n_attribute
        outputs = outputs.view(n_attribute, -1, self.d_v)
        outputs = torch.transpose(outputs, 0, 1)
        outputs = self.proj(outputs)
        return outputs

    def forward(self, feature, attribute):
        attribute = attribute.unsqueeze(-1)
        feature = feature.unsqueeze(-1)
        #TODO: linear transformation qkv with attribute
        attribute_q = self.attribute_q_proj(attribute).transpose(0, 1).unsqueeze(-2)
        attribute_k = self.attribute_k_proj(attribute).transpose(0, 1).unsqueeze(-2)
        attribute_v = self.attribute_v_proj(attribute).transpose(0, 1).unsqueeze(-2)
        attri = {'q': attribute_q, 'k': attribute_k, 'v': attribute_v}
        #feature: B x 1024 (B x feature length)
        fea_q, fea_k, fea_v = self.split_qkv(feature, attri)
        fea_q = fea_q.view(-1, 1, self.d_qk)
        fea_k = fea_k.view(-1, self.feature_input_dim, self.d_qk)
        fea_v = fea_v.view(-1, self.feature_input_dim, self.d_v)

        fea_outputs, _ = self.attention(fea_q, fea_k, fea_v)
        torch.cuda.empty_cache()
        fea_outputs = self.combine_v(fea_outputs)
        fea_outputs = self.residual_dropout(fea_outputs)
        outputs = fea_outputs + feature
        return outputs

class Tuner(nn.Module):
    def __init__(self, regressor, smpl_mean_params):
        super(Tuner, self).__init__()
        self.regressor = regressor
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.fc = nn.Linear(2205 + 24*6 + 13, 1024)
        self.drop = nn.Dropout()

        self.decpose = nn.Linear(1024, 24 * 6)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.attribute_loss_func = nn.MSELoss()
    
    def forward(self, feature, attribute, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = feature.shape[0]
       
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        reg = self.regressor(feature, attribute).squeeze(-1)
        xc = torch.cat([reg, pred_pose, pred_shape, pred_cam], 1)
        xc = self.fc(xc)
        xc = self.drop(xc)
        pred_pose = self.decpose(xc) + pred_pose
        pred_shape = self.decshape(xc) + pred_shape
        pred_cam = self.deccam(xc) + pred_cam
        # for i in range(n_iter):
        #     xc = torch.cat([reg, pred_pose, pred_shape, pred_cam],1)
        #     xc = self.fc1(xc)
        #     xc = self.drop1(xc)
        #     xc = self.fc2(xc)
        #     xc = self.drop2(xc)
        #     pred_pose = self.decpose(xc) + pred_pose
        #     pred_shape = self.decshape(xc) + pred_shape
        #     pred_cam = self.deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        
        # if self.body_measurements is not None:
        return pred_rotmat, pred_shape, pred_cam

class Measurement(nn.Module):
    def __init__(self):
        super().__init__()
        self.body_measurements = BodyMeasurements({'meas_definition_path': '/data/measurement_definitions.yaml',
                 'meas_vertices_path': '/data/smpl_measurement_vertices.yaml'})

    def forward(self, v_shaped, faces_tensor):
        shaped_triangle = v_shaped[:, faces_tensor]
        measurements = self.body_measurements(shaped_triangle)['measurements']
        meas_dict = {}
        for name, d in measurements.items():
            meas_dict[name] = d['tensor']
        return meas_dict

    def attribute_loss(self, attribute, gt_attribute):
        return self.attribute_loss_func(attribute, gt_attribute)