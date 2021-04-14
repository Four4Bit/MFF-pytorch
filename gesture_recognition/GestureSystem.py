import ctypes
import heapq
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data.sampler import SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from gesture_recognition import datasets_video
from gesture_recognition.dataset import TSNDataSet
from gesture_recognition.main import AverageMeter, accuracy
from gesture_recognition.models import TSN
from gesture_recognition.opts import parser
from gesture_recognition.transforms import *


class GestureSystem:
    """
    手势识别系统
    1. 可以加载jester数据集进行训练
    2. 可以加载jester数据集进行验证
    3. 可以处理某视频得到连续帧数据并作为一组数据进行识别
    4. 可以加载之前训练好的模型
    """

    def __init__(self, base_model_name=None):
        # 加载参数
        self.num_class, self.prefix = self._set_args()

        # 设置日志（包括输出日志和tensorboard）
        self._set_log()

        # 设置模型
        if base_model_name is not None:
            self.args.arch = base_model_name
        self.crop_size, self.input_mean, self.input_std, self.scale_size, self.train_augmentation \
            = self._set_model(self.num_class)

        # 令cudnn根据卷积网络的实际结构选择最佳的实现算法
        cudnn.benchmark = True

        # 设置损失函数
        self._set_loss_function()

        self.train_loader = None
        self.val_loader = None
        self._set_dataloader_args(self.input_mean, self.input_std)

        # 设置实时视频处理
        self.deal_video = DealVideo(self.data_length, self.args.modality, self.normalize, self.args.arch,
                                    self.scale_size, self.crop_size, self.args.num_segments)

    def __del__(self):
        self.board_writer.close()
        self.output_log.close()

    def _set_loss_function(self):
        # define loss function (criterion) and optimizer
        if self.args.loss_type == 'nll':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            raise ValueError("Unknown loss type")

    def _set_train_dataloader(self):
        # 训练数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            TSNDataSet(self.args.root_path, self.args.train_list, num_segments=self.args.num_segments,
                       new_length=self.data_length,
                       modality=self.args.modality,
                       image_tmpl=self.prefix,
                       dataset=self.args.dataset,
                       transform=torchvision.transforms.Compose([
                           self.train_augmentation,
                           Stack(roll=(self.args.arch in ['BNInception', 'InceptionV3']),
                                 isRGBFlow=(self.args.modality == 'RGBFlow')),
                           ToTorchFormatTensor(div=(self.args.arch not in ['BNInception', 'InceptionV3'])),
                           self.normalize,
                       ])),
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=False)

    def _set_val_dataloader(self):
        # 验证数据加载器
        self.val_loader = torch.utils.data.DataLoader(
            TSNDataSet(self.args.root_path, self.args.val_list, num_segments=self.args.num_segments,
                       new_length=self.data_length,
                       modality=self.args.modality,
                       image_tmpl=self.prefix,
                       dataset=self.args.dataset,
                       random_shift=False,
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(self.scale_size)),
                           GroupCenterCrop(self.crop_size),
                           Stack(roll=(self.args.arch in ['BNInception', 'InceptionV3']),
                                 isRGBFlow=(self.args.modality == 'RGBFlow')),
                           ToTorchFormatTensor(div=(self.args.arch not in ['BNInception', 'InceptionV3'])),
                           self.normalize,
                       ])),
            batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=False)

    def _set_dataloader_args(self, input_mean, input_std):
        # Data loading code
        if (self.args.modality != 'RGBDiff') | (self.args.modality != 'RGBFlow'):
            self.normalize = GroupNormalize(input_mean, input_std)
        else:
            self.normalize = IdentityTransform()
        if self.args.modality == 'RGB':
            self.data_length = 1
        elif self.args.modality in ['Flow', 'RGBDiff']:
            self.data_length = 5
        elif self.args.modality == 'RGBFlow':
            self.data_length = self.args.num_motion
        else:
            raise Exception("ars.modality is not allowed.")

    def _set_args(self):
        self.best_prec1 = 0
        self.args = parser.parse_args()  # 导入配置参数
        self._check_rootfolders()  # 创建日志和模型文件夹

        # 标签列表，训练集txt路径，验证集txt路径，数据根路径（datasets/jester），图片文件名（{:05d}.jpg）
        self.categories, self.args.train_list, self.args.val_list, self.args.root_path, prefix = \
            datasets_video.return_dataset(self.args.dataset, self.args.modality)
        num_class = len(self.categories)
        self.categories_map = {}
        for label_id, label in enumerate(self.categories):
            self.categories_map[label_id] = label

        self.args.store_name = '_'.join(['MFF', self.args.dataset, self.args.modality, self.args.arch,
                                         'segment%d' % self.args.num_segments, '%df1c' % self.args.num_motion])
        print('storing name: ' + self.args.store_name)

        return num_class, prefix

    def _set_log(self):
        # tensorboard log
        self.board_writer = SummaryWriter("./log/tensorboard")
        # 输出日志
        self.output_log = open(os.path.join(self.args.root_log, '%s.csv' % self.args.store_name), 'w')

    def _set_model(self, num_class):
        # 数据模型
        self.model = TSN(num_class, self.args.num_segments, self.args.modality,
                         base_model=self.args.arch,
                         consensus_type=self.args.consensus_type,
                         dropout=self.args.dropout, num_motion=self.args.num_motion,
                         img_feature_dim=self.args.img_feature_dim,
                         partial_bn=not self.args.no_partialbn,
                         dataset=self.args.dataset)
        crop_size = self.model.crop_size
        scale_size = self.model.scale_size
        input_mean = self.model.input_mean
        input_std = self.model.input_std
        train_augmentation = self.model.get_augmentation()

        # 根据学习率定义优化器（SGD）
        self._set_optimizer()

        # 数据模型（转化为cuda）
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpus).cuda()

        # 加载训练好的模型
        if self.args.resume:
            self._load_model()

        print(self.model)

        return crop_size, input_mean, input_std, scale_size, train_augmentation

    def _set_optimizer(self):
        # 学习率调增策略
        policies = self.model.get_optim_policies()

        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

        # 优化器
        self.optimizer = torch.optim.SGD(policies,
                                         self.args.lr,
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)

    def _load_model(self):
        if os.path.isfile(self.args.resume):
            print(("=> loading checkpoint '{}'".format(self.args.resume)))
            checkpoint = torch.load(self.args.resume)
            self.args.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {})".format(checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(self.args.resume)))

    def _train_all(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self._adjust_learning_rate(epoch, self.args.lr_steps)

            # train for one epoch
            self._train_epoch(epoch)

            # 在训练结束后评估模型，完成后退出
            # evaluate on validation set
            if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
                prec1 = self.validate()

                # remember best prec@1 and save checkpoint
                is_best = prec1 > self.best_prec1
                self.best_prec1 = max(prec1, self.best_prec1)
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best)
            else:
                # 每次存储检查点而不验证
                self._save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, False)

    def _check_rootfolders(self):
        """Create log and model folder"""
        folders_util = [self.args.root_log, self.args.root_model, self.args.root_output]
        for folder in folders_util:
            if not os.path.exists(folder):
                print('creating folder ' + folder)
                os.mkdir(folder)

    def _save_checkpoint(self, state, is_best):
        # 存在log/..._checkpoint.pth.tar中
        torch.save(state, '%s/%s_checkpoint.pth.tar' % (self.args.root_model, self.args.store_name))
        if is_best:
            shutil.copyfile('%s/%s_checkpoint.pth.tar' % (self.args.root_model, self.args.store_name),
                            '%s/%s_best.pth.tar' % (self.args.root_model, self.args.store_name))

    def _adjust_learning_rate(self, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
        lr = self.args.lr * decay
        decay = self.args.weight_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']

    def _train_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if self.args.no_partialbn:
            self.model.module.partialBN(False)
        else:
            self.model.module.partialBN(True)

        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (input_data, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            input_var = Variable(input_data)
            target_var = Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))
            top5.update(prec5.item(), input_data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()

            loss.backward()

            if self.args.clip_gradient is not None:
                total_norm = clip_grad_norm(self.model.parameters(), self.args.clip_gradient)

            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(self.train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=self.optimizer.param_groups[-1]['lr']))
                print(output)

                self.output_log.write(output + '\n')
                self.output_log.flush()

                self.board_writer.add_scalar("loss", losses.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("prec@1", top1.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("prec@5", top5.val, global_step=epoch * len(self.train_loader) + i)
                self.board_writer.add_scalar("lr", self.optimizer.param_groups[-1]['lr'],
                                             global_step=epoch * len(self.train_loader) + i)
                self.board_writer.flush()

    def print_model(self):
        """**打印模型**"""
        print(self.model)

    def validate(self):
        """**验证模型**

        Returns:
            准确率的平均值
        """
        if self.val_loader is None:
            self._set_val_dataloader()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for i, (input_data, target) in enumerate(self.val_loader):
            target = target.cuda()
            with torch.no_grad():
                input_var = Variable(input_data)
                target_var = Variable(target)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            # 测量精度并记录损失(最高值为正确的精确度，前5大值包含预测正确的精确度)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.data.item(), input_data.size(0))
            top1.update(prec1.item(), input_data.size(0))
            top5.update(prec5.item(), input_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                          .format(i, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
                print(output)
                self.output_log.write(output + '\n')
                self.output_log.flush()

        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses))
        print(output)
        output_best = '\nBest train Prec@1: %.3f' % self.best_prec1
        print(output_best)
        self.output_log.write(output + ' ' + output_best + '\n')
        self.output_log.flush()

        return top1.avg

    def load_model(self, model_path=None):
        """**加载已有模型参数**

        要求和设置的网络对应

        Args:
            model_path: 加载的目标模型路径
        """
        if model_path is not None:
            self.args.resume = model_path
        self._load_model()

    def train(self, epochs=None, epochs_start=None, eval_freq=None):
        """**训练**

        默认epochs从0训练100次

        Args:
            epochs: 训练轮次
            epochs_start: 训练开始轮次（用于多次调用该函数接着上次的训练）
            eval_freq: 验证模型与更新最佳模型频率
        """
        if epochs is not None:
            self.args.epochs = epochs
        if epochs_start is not None:
            self.args.start_epoch = epochs_start
        if eval_freq is not None:
            self.args.eval_freq = eval_freq

        if self.train_loader is None or self.val_loader is None:
            self._set_train_dataloader()
            self._set_val_dataloader()
        self._train_all()

    def classify(self, rgb_img):
        end = time.time()

        # switch to evaluate mode
        self.model.eval()

        mff_input_time = time.time()
        self.deal_video.push_new_rgb(rgb_img)
        mff_input = self.deal_video.get_mff_input_from_buffer()
        print("calculate mff input use time: {}".format(time.time() - mff_input_time))

        if mff_input is None:
            return None
        with torch.no_grad():
            input_var = Variable(mff_input)

        # compute output
        forward_time = time.time()
        output = self.model(input_var)
        print("forward use time: {}".format(time.time() - forward_time))

        # measure elapsed time
        print("classify: Time: {:.3f}".format(time.time() - end))

        # top = output.topk(1, dim=1, largest=True, sorted=True)
        # return (top.indices.item(), top.values.item())
        # return max(enumerate(output[0]), key=lambda x: x[1])
        return heapq.nlargest(5, enumerate(output[0]), key=lambda x: x[1])


class DealVideo:
    def __init__(self, new_length, modality, normalize, model_name, scale_size, crop_size,
                 num_segments, transform=None):
        self.new_length = new_length
        self.modality = modality
        self.normalize = normalize
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.num_segments = num_segments
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([
                GroupScale(int(self.scale_size)),
                GroupCenterCrop(self.crop_size),
                Stack(roll=(model_name in ['BNInception', 'InceptionV3']),
                      isRGBFlow=(self.modality == 'RGBFlow')),
                ToTorchFormatTensor(div=(model_name not in ['BNInception', 'InceptionV3'])),
                self.normalize,
            ])

        if self.modality == 'RGBDiff' or self.modality == 'RGBFlow':
            self.new_length += 1  # Diff needs one more image to calculate diff

        # 加载计算光流dll
        self.dll_path = "./lib/flow_computation.dll"
        self.lib = ctypes.CDLL(self.dll_path, winmode=ctypes.RTLD_GLOBAL)
        self.lib.calulateFlow_kkk.restype = ctypes.POINTER(ctypes.c_uint8)

        # 加载计算DeepFlow计算器
        self.inst = cv2.optflow.createOptFlow_DeepFlow()

        self.min_num = self.new_length + self.num_segments - 1

        # 初始化图片缓冲区
        self.buffer_length = 24
        self.rgb_list = []
        self.gray_list = []
        self.flow_u_list = []
        self.flow_v_list = []

    @staticmethod
    def _get_val_indices(num_frames, num_segments, new_length):
        if num_frames > num_segments + new_length - 1:
            tick = (num_frames - new_length + 1) / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
        return offsets + 1

    def set_buffer_length(self, length):
        self.buffer_length = length
        self.update_buffer()

    def push_new_rgb(self, rgb_img):
        self.rgb_list.append(rgb_img)
        self.gray_list.append(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY))
        if len(self.rgb_list) > 1:
            flow_u, flow_v = self.calculate_brox_flow_with_dll(self.gray_list[-2], self.gray_list[-1])
            self.flow_u_list.append(flow_u)
            self.flow_v_list.append(flow_v)
        self.update_buffer()

    def update_buffer(self):
        while len(self.rgb_list) > self.buffer_length:
            self.rgb_list.pop(0)
            self.gray_list.pop(0)
        while len(self.flow_u_list) > self.buffer_length - 1:
            self.flow_u_list.pop(0)
            self.flow_v_list.pop(0)

    def get_mff_input_from_buffer(self):
        if len(self.rgb_list) < self.min_num:
            return None
        indices = DealVideo._get_val_indices(len(self.rgb_list), self.num_segments, self.new_length)
        return self._turn_imgs_to_mff_input(self.rgb_list, self.flow_u_list, self.flow_v_list, indices)

    def turn_imgs_to_mff_input(self, rgb_list):
        indices = DealVideo._get_val_indices(len(rgb_list), self.num_segments, self.new_length)
        flow_u_list, flow_v_list = self.calculate_rgb_list_flow(rgb_list)
        return self._turn_imgs_to_mff_input(rgb_list, flow_u_list, flow_v_list, indices)

    def calculate_rgb_list_flow(self, rgb_list, mode="brox"):
        gray_list = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb_list]
        flow_u_list = []
        flow_v_list = []
        for i in range(len(gray_list) - 1):
            if mode == "brox":
                flow_u, flow_v = self.calculate_brox_flow_with_dll(gray_list[i], gray_list[i + 1])
            elif mode == "deep_flow":
                flow_u, flow_v = self.calculate_deep_flow(gray_list[i], gray_list[i + 1])
            else:
                raise Exception("calculate flow mode is not allowed")
            flow_u_list.append(flow_u)
            flow_v_list.append(flow_v)
        return flow_u_list, flow_v_list

    def _turn_imgs_to_mff_input(self, rgb_list, flow_u_list, flow_v_list, indices):
        """**将连续n帧图片数据转化为对应采样与数据融合的输入数据**

        Args:
            rgb_list: rgb图片，PIL.Image.Image类型
            flow_u_list: flow_u图片，PIL.Image.Image类型
            flow_v_list: flow_v图片，PIL.Image.Image类型
            indices: 采样索引
        """
        indices = indices - 1
        num_frames = len(rgb_list)
        images = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.modality == 'RGBFlow':
                    if i == self.new_length - 1:
                        seg_imgs = [Image.fromarray(rgb_list[p]).convert('RGB')]
                    else:
                        if p == num_frames:
                            seg_imgs = [Image.fromarray(flow_u_list[p - 1]).convert('L'),
                                        Image.fromarray(flow_v_list[p - 1]).convert('L')]
                        else:
                            seg_imgs = [Image.fromarray(flow_u_list[p]).convert('L'),
                                        Image.fromarray(flow_v_list[p]).convert('L')]
                else:
                    raise Exception("self.modality {} is not allowed".format(self.modality))

                images.extend(seg_imgs)
                if p < num_frames:
                    p += 1

        # images 一组图片共（self.new_length - 1）*2 + 1张，前self.new_length - 1每次对应横纵两张光流图，最后一张对应rgb；共self.num_segments组
        process_data = self.transform(images)
        return process_data

    def calculate_brox_flow_with_dll(self, frame1, frame2):
        """调用c++ dll计算brox光流

        Args:
            frame1: 第一张图片（灰度图）
            frame2: 第二张图片（灰度图）
        """
        start_time = time.time()

        # 转化np为uchar*
        frame_data1 = np.asarray(frame1, dtype=np.uint8)
        frame_data1 = frame_data1.ctypes.data_as(ctypes.c_char_p)
        frame_data2 = np.asarray(frame2, dtype=np.uint8)
        frame_data2 = frame_data2.ctypes.data_as(ctypes.c_char_p)

        p = self.lib.calulateFlow_kkk(frame1.shape[0], frame1.shape[1], frame_data1,
                                      frame2.shape[0], frame2.shape[1], frame_data2)

        # 转化uchar*为np
        ret = np.array(np.fromiter(p, dtype=np.uint8,
                                   count=frame1.shape[0] * frame1.shape[1] + frame2.shape[0] * frame2.shape[1]))
        ret_u = ret[:frame1.shape[0] * frame1.shape[1]].reshape(frame1.shape)
        ret_v = ret[frame1.shape[0] * frame1.shape[1]:].reshape(frame2.shape)

        self.lib.release_kkk(p)

        print("calculate flow use time: {}s".format(time.time() - start_time))

        return ret_u, ret_v

    def calculate_deep_flow(self, frame1, frame2):
        start_time = time.time()

        flow = self.inst.calc(frame1, frame2, None)

        print("calculate flow use time: {}s".format(time.time() - start_time))

        return flow[:, :, 0], flow[:, :, 1]


def main():
    gesture_sys = GestureSystem("resnet101")
    gesture_sys.load_model("./model/cc_MFF_jester_RGBFlow_resnet101_segment5_3f1c_best.pth.tar")

    step = 1
    labels = ["click", "turn", "zone", "catch", "swipe"]
    labels_i = 0
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if success:
            # if step % 2 == 0:
            if True:
                result = gesture_sys.classify(frame_rgb)
                if result is not None:
                    # print(gesture_sys.categories_map[result[0]])
                    for line in result:
                        print(gesture_sys.categories_map[line[0]], end=", ")
                    print()
                    print()
            else:
                gesture_sys.deal_video.push_new_rgb(frame_rgb)
            if step % 40 == 0:
                print("*******######## {} ######*********".format(labels[labels_i]))
                labels_i += 1
                print()
            step += 1
            cv2.imshow("show", frame)
            cv2.waitKey(5)
        else:
            print("video read error")
            return


if __name__ == '__main__':
    main()
