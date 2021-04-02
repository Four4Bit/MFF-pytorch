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
# new_length与输入数据类型相关;不同输入snippet的融合方式:avg; ; ;dropout参数:缓解过拟合;img_feature_dim特征维度;
# 第几张图片特征图显示出来
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
        self.reshape = True  # 是否reshape
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.dataset = dataset
        self.crop_num = crop_num # 裁剪数量
        self.consensus_type = consensus_type # 聚集函数G的设置，avg,max,topk,cnn,rnn
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
            self.consensus = ConsensusModule(consensus_type) # consensusModule是聚集函数模块定义的关键

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    # 修改最后一层全连接层，并返回最后一层的输入特征图的channel数（RGB为3，单色图为1，每个卷积层中卷积核的数量），resnet和BNInception的last_layer_name为fc，
    # mobilenetv2的为classifier，若dropout不为0则添加Dropout层和新的全连接层，否则直接修改最后一层全连接层，对全连接层的权重和偏置进行初始化
    """
    _prepare_tsn方法。feature_dim是网络最后一层的输入feature map的channel数。
    接下来如果有dropout层，那么添加一个dropout层后连一个全连接层，否则就直接连一个全连接层。
    setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，一般可以用来修改网络结构，
    以setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))为例，输入包含3个值，分别是基础网络，要赋值的属性名，要赋的值，
    一般而言setattr的用法都是这样。因此当这个setattr语句运行结束后，self.base_model.last_layer_name这一层就是nn.Dropout(p=self.dropout)。 
    最后对全连接层的参数（weight）做一个0均值且指定标准差的初始化操作，参数（bias）初始化为0。getattr同样是torch.nn.Module类的一个方法，
    与为属性赋值方法setattr相比，getattr是获得属性值，一般可以用来获取网络结构相关的信息，以getattr(self.base_model, self.base_model.last_layer_name)为例，
    输入包含2个值，分别是基础网络和要获取值的属性名。
    """
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
            # 调整input_size,mean,std为适合网络的值

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
                        # eval()将字符串转换成表达式计算

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        # 梯度需求置否，使更新停止

    def partialBN(self, enable):
        self._enable_pbn = enable

    # get_optim_policies对第一个卷积层、全连接层、bn层、普通层操作设置不同，偏置的乘子都为0表示不需要学习
    # 提取各层参数
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
                ps = list(m.parameters())   # 将参数转换成list保存进ps
                conv_cnt += 1   # 计数卷积层数量，出现2d或1d卷积就计一次
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0]) # 提取第一个卷积层的weight
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])   # 如果参数是2，即bias不是0，取bias
                else:
                    normal_weight.append(ps[0]) # 取各层参数
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):    # 同上，这是全连接层（线性变换层）
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters())) # extend()用于在list末尾追加另一个序列的多个值
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

    def forward(self, input): # TSN类的forward函数定义了模型前向计算过程，也就是TSN的base_model+consensus结构
        # RGB为3*new_length,其他为2*new_length
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)   # RGB过一遍_get_diff函数，得到RGBDiff的input

        if self.modality == 'RGBFlow':
            sample_len = 3 + 2 * self.new_length

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0: # 是否加入dropout
            base_out = self.new_fc(base_out)

        if not self.before_softmax: # 如果未加softmax，加softmax
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # tensor.view(-1,3)为将base_out维度重置为(-1,3)的tensor,-1的位置根据实际size改变，实际为原size积/3，比如原来为(6,9),view(-1,3)后变为(18,3)
        # size()[1:]返回第二个维度之后的维度(包括第二个维度自身)，比如(10,3,4,5)，就返回(3,4,5)

        output = self.consensus(base_out)   # 经过consensus函数，得到最终输出
        return output.squeeze(1)    # squeeze(1)与squeeze()一个作用，去掉tensor中为1的维度，比如(10,1,2,1)变为(10,2)

    def _get_diff(self, input, keep_rgb=False):     # RGB，RGBDiff为3，Flow为2，此处input_c是通道数
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        # 具体size较复杂，一般为input.view((-1,3,2,3)+input.size()[2:])

        if keep_rgb:
            new_data = input_view.clone()   # c lone()与copy()一样，复制一个tensor，开辟一块新的内存给这个tensor
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()
        # 复制一份input_view,其中第三个维度取第二之后的维度,比如(10,1,2,3,4,5)[:,:,1:,:,:,:]即为(10,1,1,3,4,5)
        # 也就是第三个维度是通道数，keep_rgb即为3，否则减1变为2，即flow的通道数

        for x in reversed(list(range(1, self.new_length + 1))): # 倒着取，也就是[2,1]
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                # 求RGBDiff,R-B,G-R,B-G作为新的RGB
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                # Flow差值,一般的Flow是(h色调,s饱和度,v亮度)三通道的，这个网络提取时只提取了(s，v)双通道的Flow?

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
        # lambda为匿名函数，这里表示参数为x，isinstance判断modules中有多少卷积层,取第一个作为第一个卷积层的index
        # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。（判断函数，可迭代对象）
        # 接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]    # 定义一个卷积层，值为原来的第一个卷积层
        container = modules[first_conv_idx - 1] # 定义原来第一个卷积层前的一层为container

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()] # 克隆卷积层参数
        kernel_size = params[0].size()  # params[0]是weight
        print(kernel_size)
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:] # 卷积核大小的设置，下面注释有说
        print(new_kernel_size)
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous() # contiguous()保证内存连续性，使得view()函数可以正常使用

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False) # 重新设置卷积层
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    """
       前面提到如果输入不是RGB，那么就要修改网络结构，这里以models.py脚本中TSN类的_construct_flow_model方法介绍对于optical flow类型的输入需要修改哪些网络结构。
       conv_layer是第一个卷积层的内容，params 包含weight和bias，kernel_size就是(64,3,7,7)，
       因为对于optical flow的输入，self.new_length设置为5，所以new_kernel_size是(63,10,7,7)。new_kernels是修改channel后的卷积核参数，主要是将原来的卷积核参数复制到新的卷积核。
       然后通过nn.Conv2d来重新构建卷积层。new_conv.weight.data = new_kernels是赋值过程。
    """

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
