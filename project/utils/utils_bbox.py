import numpy as np
import os
import torch
from torchvision.ops import nms
import pkg_resources as pkg
import config


def check_version(
    current: str = "0.0.0",
    minimum: str = "0.0.0",
    name: str = "version",
    pinned: bool = False,
) -> bool:
    current, minimum = (pkg.parse_version(str(x)) for x in (current, minimum))  # type: ignore
    result = (current == minimum) if pinned else (current >= minimum)
    return result


TORCH_2_31 = check_version(current=torch.__version__, minimum="2.3.1")


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate Anchors from Features."""
    anchor_points, stride_tensors = [], []  # 用于存储生成的锚点和步幅张量

    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):  # 对每一个图与其步幅张量进行遍历
        _, _, h, w = feats[i].shape  # 获取特征值的宽高

        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x    # 生成一系列x轴坐标，范围从0-w，计算时增加上偏置
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y    # 同x

        sy, sx = (
            torch.meshgrid(sy, sx, indexing="ij")
            if TORCH_2_31
            else torch.meshgrid(sy, sx)
        )  # meshgrid可以经过输入两个值的参数来生成网格坐标
        # 具体是作笛卡尔积，sy是每一行相同，每一列代表y方向上的坐标值

        # 将张量堆叠为2D，并展平存入锚点列表
        # stack方法能使两个张量在一个维度上堆叠，-1表示在最后一个维度
        # view()方法能重塑张量，-1表示由方法自行判断，2表示最终生成一个2维张量，形状为[h*w, 2]
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))

        # full方法会生成一个形状[w*h, 1]的一维张量，每个值都为该图片的stride张量
        stride_tensors.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensors)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance (ltrb) to box  (xywh)"""
    """
        该方法输入一个distance表示相对于anchor_points的边界框位置
        将其转换为用左上右下坐标点方式表示边界框
        如果需要使用xywh格式，则进一步计算两个点的坐标的相对距离，生成w和h
    """
    # 左上右下
    # anchor_points保存着锚点的二维坐标
    # distance保存着左上和右下分别的二维距离信息[ltx, lty, rbx, rby]
    # split方法将distance按步长2切割，即两两取出
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt  # [ltx, lty]
    x2y2 = anchor_points + rb  # [rbx, rby]
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class DecodeBox:
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.bbox_attrs = 4 + num_classes

    def decode_box(self, inputs):
        # 边界框信息、类别、原始类别、锚点信息、步幅信息
        dbox, cls, origin_cls, anchors, strides = inputs

        # 获得中心点宽坐标
        dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides

        # cls.sigmoid()对cls进行激活，返回每个类别的概率，形状是一个二维张量
        # cat方法将边界框信息和分类结果（类别和概率）拼接，并通过permute方法进行维度变换
        y = torch.cat((dbox, cls.sigmoid()), dim=1).permute(0, 2, 1)

        # 进行归一化，到0~1之间
        y[:, :, :4] = y[:, :, :4] / torch.Tensor(
            [
                self.input_shape[1],
                self.input_shape[0],
                self.input_shape[1],
                self.input_shape[0],
            ]
        ).to(
            y.device()
        )  # type: ignore
        return y

    # yolo_current_boxes方法
    # 该方法用于将解码后的边界框坐标从模型输入调整到原始图像尺寸，并考虑到了图像预处理时可能进行的填充(letterboxing)操作
    # 它执行步骤：
    # a.交换边界框坐标和y轴和x轴，以便和图像的宽高对应
    # b.如果在图像预处理时进行了填充，计算填充后的图像相对于原始图像的偏移量和缩放比例尽量保持一致
    # c.计算边界框和最小和最大坐标，并将它们拼成最终的边界框格式，[xmin,ymin,xmax,ymax]
    # d.将边界框坐标从模型输入尺寸缩放到原始图像尺寸
    def yolo_current_boxes(
        self, box_xy, box_wh, input_shape, image_shape, letterbox_image
    ):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # round()方法用于四舍五入，第二参数为位数，默认0即整数
            # 缩放比例取最小比值，确保新图像尺寸能够覆盖整个原始图像
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (
                (input_shape - new_shape) / 2.0 / input_shape
            )  # 将尺寸差值比例求取并除以2计算出填充的比例
            scale = input_shape / new_shape  # 计算得到比例

            box_yx = (
                box_yx - offset
            ) * scale  # 减去偏移量，乘以比例从而将经模型输出的坐标移动回原始图像
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)
        boxes = np.concatenate(
            [
                box_mins[..., 0:1],
                box_mins[..., 1:2],
                box_maxes[..., 0:1],
                box_maxes[..., 1:2],
            ],
            axis=-1,
        )
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    # nms
    def non_max_suppression(
        self,
        prediction, # (batch_size, num_boxes, 5 + num_classes)
        num_classes,
        input_shape,
        image_shape,
        letterbox_image,
        conf_thres=0.5,
        nms_thres=0.4,
    ):
        box_corner = prediction.new(prediction.shape)   # 浅拷贝
        # 将中心点坐标转化为角点坐标
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4] 

        # 初始化一个空的输出列表
        output = [None for _ in range(len(prediction))]

        # 对每张图片的边界框进行提取，过滤掉置信度低于阈值的边界框
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(
                image_pred[:, 4 : 4 + num_classes], 1, keepdim=True
            )

            # 置信度需大于阈值
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            
            if not image_pred.size(0):
                continue

            detections = torch.cat(
                (image_pred[:, :4], class_conf.float(), class_pred.float()), 1
            )

            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(detections_class[:, :4], detections_class[:, 4], nms_thres)
                max_detections = detections_class[keep]

                # 首先检查output[i] 是否为 None，一般在处理第一张图像时，因为在之前咱们初始化的时候设置为是 None
                # 如果output[i] 是 None（即：当前处理的第一张图像，或者之前图像没有检测到，没有任何目标）直接向
                output = (
                    max_detections
                    if output[i] is None
                    else torch.cat((output[i], max_detections))  # type: ignore
                )

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()  # type: ignore
                # [:, 0:2] 选择张量的所有行和前两列（0:2表示从索引0开始到索引1结束，即第一列和第二列）
                box_xy, box_wh = (
                    output[i][:, 0:2] + output[i][:, 2:4] / 2,  # type: ignore
                    output[i][:, 2:4] - output[i][:, 0:2],  # type: ignore
                )

                output[i][:, :4] = self.yolo_current_boxes(  # type: ignore
                    box_xy, box_wh, input_shape, image_shape, letterbox_image
                )
                return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):

        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width

        scaled_anchors = [
            (anchor_width / stride_w, anchor_height / stride_h)
            for anchor_width, anchor_height in anchors[anchors_mask[2]]
        ]

        prediction = (
            input.view(
                batch_size,
                len(anchors_mask[2]),
                num_classes + 5,
                input_height,
                input_width,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = (
            torch.linspace(0, input_width - 1, input_width)
            .repeat(input_height, 1)
            .repeat(batch_size * len(anchors_mask[2]), 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )  # type: ignore
        grid_y = (
            torch.linspace(0, input_height - 1, input_height)
            .repeat(input_width, 1)
            .t()
            .repeat(batch_size * len(anchors_mask[2]), 1, 1)
            .view(y.shape)
            .type(FloatTensor)
        )  # type: ignore

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = (
            anchor_w.repeat(batch_size, 1)
            .repeat(1, 1, input_height * input_width)
            .view(w.shape)
        )
        anchor_h = (
            anchor_h.repeat(batch_size, 1)
            .repeat(1, 1, input_height * input_width)
            .view(h.shape)
        )

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2.0 - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2.0 - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32

        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32

        # 将网格的x坐标和y坐标转换为我们的numPy数组后乘以 * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32

        # 将锚框的宽度和高度转换为numPy数组后乘以32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        # 创建图形对象
        ax = fig.add_subplot(121)
        # 添加一个子图，1行2列中的第1个位置
        img = Image.open(os.path.join(config.vocPath, "JPEGImages/000003.jpg")).resize(
            [640, 640]
        )
        # 显示图像，设置透明度为0.5
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)

        # 设置x轴和y轴的显示范围，注意这里的范围可能超出了图像的实际的尺寸，用于可视化网络
        plt.scatter(grid_x, grid_y)
        # 绘制特定点(point_h, point_w)在网络中的位置，这里假设要想特殊标注的点
        plt.scatter(point_h * 32, point_w * 32, c="black")
        # 翻转y轴，因为matplotlib的y轴默认是
        plt.gca().invert_yaxis()

        # 计算锚框的左上角坐标
        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        # 创建一个矩形对rect1，用于在图像绘制
        # 矩形的左下角坐标(anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w])
        # 矩形的宽度是anchor_w[0, 0, point_h point_w]
        # 矩形的高度是anchor_h[0, 0, point_h, point_w]
        rect1 = plt.Rectangle(
            [anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]],
            anchor_w[0, 0, point_h, point_w],
            anchor_h[0, 0, point_h, point_w],
            color="r",
            fill=False,
        )
        rect2 = plt.Rectangle(
            [anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]],
            anchor_w[0, 1, point_h, point_w],
            anchor_h[0, 1, point_h, point_w],
            color="r",
            fill=False,
        )
        rect3 = plt.Rectangle(
            [anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]],
            anchor_w[0, 2, point_h, point_w],
            anchor_h[0, 2, point_h, point_w],
            color="r",
            fill=False,
        )

        # 将锚框添加到子图中
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # 添加第二个子图
        ax = fig.add_subplot(122)
        # 再次显示图像和设置坐标轴范围(0.5透明度)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)

        # 绘制网络点和特定点
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c="black")
        # box_xy包含着预测边界框的中心点坐标，这里绘制这些边框
        plt.scatter(
            box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c="r"
        )
        plt.gca().invert_yaxis()

        pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
        pre_top = box_xy[..., 1] - box_wh[..., 1] / 2

        # 第一个参数pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]矩形左下角坐标
        # box_wh[0, 0, point_h, point_w], box_wh[0, 0, point_h, point_w]矩形的宽度和高度
        rect1 = plt.Rectangle(
            [pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],
            box_wh[0, 0, point_h, point_w, 0],
            box_wh[0, 0, point_h, point_w, 1],
            color="r",
            fill=False,
        )
        rect2 = plt.Rectangle(
            [pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],
            box_wh[0, 1, point_h, point_w, 0],
            box_wh[0, 1, point_h, point_w, 1],
            color="r",
            fill=False,
        )
        rect3 = plt.Rectangle(
            [pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],
            box_wh[0, 2, point_h, point_w, 0],
            box_wh[0, 2, point_h, point_w, 1],
            color="r",
            fill=False,
        )

        # 将预测边界框添加到子图中
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # 显示图形1
        plt.show()

    # batch_size 批次大小
    # 255：通道数(channels)，在实际应用中，这个数字与模型设计有关，在Yolo模型中，通常表示特征图的宽度

    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()

    anchors = np.array(
        [
            [116, 90],
            [156, 198],
            [373, 326],
            [30, 61],
            [62, 45],
            [59, 119],
            [10, 13],
            [16, 30],
            [33, 23],
        ]
    )

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
