from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import pretrainedmodels
import MLPmodule

# tsn参数、rgb和flow的融合、new_conv = nn.conv2d函数、三个处理函数、下载model的修改和添加
# 类似代码及解释https://blog.csdn.net/u014380165/article/details/79058147

# TSN(temporal segment network)
# self;不同数据集的子类数;一个video分成多少份;输入模式:RGB，optical flow,RGB flow;采用那种模型:resnet101,BNInception;
# new_length与输入数据类型相关;不同输入snippet的融合方式:avg; ; ;dropout参数;img_feature_dim特征维度;
# TSN调用_prepare_base_model(self, base_model)和_prepare_tsn(self, num_class)完成初始化
class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True, num_motion=3,
                 dropout=0.8, img_feature_dim=256, dataset='jester',
                 crop_num=1, partial_bn=True, print_spec=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.num_motion = num_motion
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.dataset = dataset
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")
        # 对于非RGB类型我们要获取连续六个片段
        if new_length is None:
            if modality == "RGB":
                self.new_length = 1
            elif modality == "Flow":
                self.new_length = 5
            elif modality == "RGBFlow":
                # self.new_length = 1
                self.new_length = self.num_motion
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout,
                       self.img_feature_dim)))

        # 准备模型
        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)


        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        elif self.modality == 'RGBFlow':
            print("Converting the ImageNet model to RGB+Flow init model")
            self.base_model = self._construct_rgbflow_model(self.base_model)
            print("Done. RGBFlow model ready.")
        if consensus_type == 'MLP':
            self.consensus = MLPmodule.return_MLP(consensus_type, self.img_feature_dim, self.num_segments, num_class)
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    # 修改最后一层全连接层，并返回最后一层的输入特征图的channel数（RGB为3，单色图为1，每个卷积层中卷积核的数量），resnet和BNInception的last_layer_name为fc，
    # mobilenetv2的为classifier，若dropout不为0则添加Dropout层和新的全连接层，否则直接修改最后一层全连接层，对全连接层的权重和偏置进行初始化
    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            # 基础网络，要赋值的属性名，赋的值
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type == 'MLP':
                # set the MFFs feature dimension
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    # _prepare_base_model方法来导入模型，根据is_shift和non_local参数添加相应模块，base_model非resnet时无non_local，
    # 根据modality设置input_mean和input_std，resnet调用temporal_shift.py、non_local.py完成设置，
    # RGB的input_mean和input_std与imagenet一致；Flow的input_mean为0.5，input_std为RGB的input_std的均值；RGBDiff相减产生，
    # input_mean和input_std长度共num_length+1，头为RGB的值，input_mean的尾为0，input_std的尾为input_std的均值乘以2
    # mobilenetv2调用temporal_shift.py完成设置，只对expand_ratio不为1并且使用残差结构的倒置残差模块设置，input_mean和input_std与resnet类似
    # BNInception调用bn_inception.py完成模型搭建，其中build_temporal_ops()设置时序位移模块，注意到这里的input_mean非归一化，且input_std为1
    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model or 'squeezenet1_1' in base_model:
            self.base_model = pretrainedmodels.__dict__[base_model](num_classes=1000, pretrained='imagenet')
            if base_model == 'squeezenet1_1':
                self.base_model = self.base_model.features
                self.base_model.last_layer_name = '12'
            else:
                self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        # BNInception:batch normalization inception,BN使用较大的学习率去训练网络，加速网络训练;降低过拟合现象
        elif base_model == 'BNInception':
            # 此处若无模型会从官网下载预训练模型
            self.base_model = pretrainedmodels.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif 'resnext101' in base_model:
            self.base_model = pretrainedmodels.__dict__[base_model](num_classes=1000, pretrained='imagenet')
            print(self.base_model)
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    # get_optim_policies对第一个卷积层、全连接层、bn层、普通层操作设置不同，偏置的乘子都为0表示不需要学习
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        if self.modality == 'RGBFlow':
            sample_len = 3 + 2 * self.new_length

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    """ # There is no need now!!
    def _get_rgbflow(self, input):
        input_c = 3 + 2 * self.new_length # 3 is rgb channels, and 2 is coming for x & y channels of opt.flow
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        new_data = input_view.clone()
        return new_data
    """

    def _construct_rgbflow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        filter_conv2d = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))
        first_conv_idx = next(filter_conv2d)
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = torch.cat(
            (params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous(), params[0].data),
            1)  # NOTE: Concatanating might be other way around. Check it!
        new_kernel_size = kernel_size[:1] + (3 + 2 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    # 修改模型适合光流输入，修改第一个卷积层，修改输入channel，卷积核用预训练的卷积核均值赋值
    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        print(kernel_size)
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        print(new_kernel_size)
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    # 修改模型适合RGBDiff输入，与上类似，注意的是有keep_rgb参数，表示是否保留RGB卷积核参数，比较好奇的是前面num_length + 1
    # 有何用呢？
    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        # 上面告诉我们basemodel的返回卷积层，往往都是
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'RGBFlow':
            return torchvision.transforms.Compose([GroupMultiScaleResize(0.2),
                                                   GroupMultiScaleRotate(20),
                                                   # GroupSpatialElasticDisplacement(),
                                                   GroupMultiScaleCrop(self.input_size,
                                                                       [1, .875,
                                                                        .75,
                                                                        .66]),
                                                   # GroupRandomHorizontalFlip(is_flow=False)
                                                   ])
