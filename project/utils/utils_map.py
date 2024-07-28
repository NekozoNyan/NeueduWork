import glob
import json
import math
import operator
import os
import shutil
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        对数平均漏报率：
        通过平均9个等距FPPI点的漏检率计算
        在日志空间中在10e-2和10e0之间。

        输出：
        lamr|对数平均漏报率
        错过率
        fppi |每张图像的误报率
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


"""
 抛出错误并退出
"""


def error(msg):
    print(msg)
    sys.exit(0)


"""
 检查该数字是否为介于0.0和1.0之间的浮点数
"""


# 尝试转换：首先，函数尝试将传入的value参数转换为浮点数。这是通过float(value)实现的。如果value可以成功转换为浮点数，那么转换的结果会被赋值给变量val。
# 检查范围：接着，函数检查转换后的浮点数val是否大于0.0且小于1.0。这里，0.0表示浮点数0，而1.0表示浮点数1。如果val满足这个条件（即，它是一个大于0且小于1的浮点数），则函数返回True，表示value确实是一个介于0和1之间的浮点数。
# 异常处理：如果在尝试将value转换为浮点数时发生错误（例如，如果value是一个字符串，但它不是有效的数字表示），则会抛出ValueError异常。在try块之后，except ValueError块会捕获这个异常，并导致函数返回False。这表示value不是一个有效的浮点数，或者它不在0和1之间（尽管在这个特定的上下文中，后者实际上已经被前面的条件检查排除了）
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        # 0.0和1.0分别代表浮点数0和浮点数1。这个函数的目的是检查传入的value参数是否可以转换为浮点数，并且这个浮点数是否在0和1之间（不包括0和1）
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 根据召回率和精确度数组计算AP
    1）我们计算一个版本的测量精度/召回率曲线
    精度单调递减
    2）我们通过数值积分将AP计算为该曲线下的面积。
"""


# 计算目标检测或分类任务中的平均精度（Average Precision, AP）
# 参数：
# rec: 召回率列表列表（或数组），表示召回率（Recall）值。召回率是正确识别的正样本数除以所有正样本数的比例。
# prec: 精度列表列表（或数组），表示精度（Precision）值。精度是正确识别的正样本数除以所有被识别为正样本的数的比例
def voc_ap(rec, prec):
    """
    ---官方matlab代码VOC2012---
        mrec=[0；rec；1]；
        mpre=[0；prec；0]；
        i=数值（mpre）-1：1:1
        mpre（i）=最大值（mpre（i），mpre（i+1））；
        结束
        i=查找（mrec（2:结束）~=mrec（1:结束1））+1；
        ap=总和（（mrec（i）-mrec（i-1））*mpre（i））；
    """
    # 在列表rec的开头插入浮点数0.0
    rec.insert(0, 0.0)  # # 作用：修改rec列表，在列表的开始位置插入0.0
    # 在列表rec的末尾添加浮点数1.0
    rec.append(1.0)  # 作用：在rec列表的末尾添加1.0
    # 创建一个rec列表的浅拷贝，赋值给mrec
    mrec = rec[:]  # 作用：创建一个rec列表的副本（浅拷贝），赋值给变量mrec
    # 在列表prec的开头插入浮点数0.0，表示精度的起始值为0 list
    prec.insert(0, 0.0)  # 在prec列表的开始位置插入0.0  list
    # 在列表prec的末尾添加浮点数0.0，这是为了计算方便，实际计算时不会用到最后一个0
    prec.append(0.0)  # # 在prec列表的末尾添加0.0 list
    mpre = prec[:]  # 创建一个prec列表的浅拷贝，赋值给mpre，用于后续确保精度单调递减的修改
    """
     这部分使精度单调递减
        （从头至尾）
        matlab：对于i=numel（mpre）-1:-1:1
        mpre（i）=最大值（mpre（i），mpre（i+1））；
    """
    for i in range(len(mpre) - 2, -1, -1):  # 从倒数第二个元素开始向前遍历
        mpre[i] = max(mpre[i], mpre[i + 1])  # 确保当前位置的精度值不小于下一个位置的精度值
    """
     这部分创建了一个召回更改的索引列表
		matlab:i=find（mrec（2:结束）~=mrec（1:结束1））+1；
    """
    i_list = []
    for i in range(1, len(mrec)):  # 遍历mrec列表（从第二个元素开始）
        if mrec[i] != mrec[i - 1]:  # 如果当前召回率与前一个召回率不同
            i_list.append(i)  # if it was matlab would be i + 1 # 将当前索引添加到列表中（注意MATLAB中索引从1开始，Python从0开始）
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    # ap -- 平均精度值
    # mrec -- 修改后的召回率列表（包含边界值）
    # mpre -- 修改后的精度列表（包含边界值并确保单调递减）
    ap = 0.0
    for i in i_list:
        # mrec[i]-mrec[i-1]计算的是相邻两个召回率值之间的差，这个差值表示了召回率的变化量。由于mrec列表是单调递增的，并且i_list中存储的是召回率发生变化的索引，所以这个差值实际上就是当前召回率区间的宽度。
        # mpre[i]表示在召回率为mrec[i]时的最大精度值（由于之前已经对mpre进行了处理，确保了它是单调递减的，所以可以用mpre[i]来近似表示整个召回率区间内的精度值）。
        # (mrec[i]-mrec[i-1])*mpre[i]计算的就是召回率从mrec[i-1]变化到mrec[i]时，对应的精度-召回率曲线下的梯形面积。这里假设在召回率区间[mrec[i-1], mrec[i]]内，精度值保持不变，等于mpre[i]。
        # 通过遍历i_list并对每个召回率区间的梯形面积进行累加，最终得到的就是近似的AP值。
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    # 使用with语句打开文件，确保文件在操作完成后正确关闭
    # path参数是文件的路径
    with open(path) as f:
        # 读取文件的所有行到content变量中，readlines方法会保留行尾的换行符
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    # 使用列表推导式遍历content中的每一行，并去除每行末尾的空白字符（包括换行符'\n'）
    # strip方法会移除字符串开头和结尾的空白字符
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
    # 设置字体样式为HERSHEY_PLAIN，这是OpenCV提供的一种基本字体样式
    font = cv2.FONT_HERSHEY_PLAIN
    # 设置字体缩放比例，这里设置为1表示原始大小
    fontScale = 1
    # 设置线条类型，这里使用cv2.LINE_AA（抗锯齿线型），但在函数调用中未直接使用，但默认为1也相当于LINE_8
    lineType = 1
    # 文本绘制的起始位置，即左下角坐标
    bottomLeftCornerOfText = pos
    # 在图像上绘制文本
    # 参数依次为：图像，文本内容，文本左下角坐标，字体，字体缩放比例，文本颜色，线条类型
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    # 获取文本的宽度和高度（返回的是一个元组，第一个元素是宽度，第二个元素是高度）
    # 但这里我们只关心宽度，因此通过[0]索引获取宽度，并忽略高度
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    # 函数返回修改后的图像和文本宽度加上额外指定的线宽（可能是为了某种布局计算）
    # 注意：这里的返回逻辑可能并不完全符合通常的用途，因为直接返回线宽+文本宽可能不够直观
    # 但在某些特定布局计算中可能有用
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


# adjust_axes 的主要目的是尝试通过调整 x 轴的显示范围来确保图表中的文本标签完全可见
# r: 渲染器对象（通常是 matplotlib.backends.backend_agg.RendererAgg 的一个实例）。这个参数用于获取文本标签的边界框（bounding box），因为文本标签的最终大小和位置在渲染时才能确定。
# t: matplotlib.text.Text 对象，代表图表中要显示的文本标签。
# fig: matplotlib.figure.Figure 对象，代表整个图表。这个参数用于获取图表的当前宽度（以英寸为单位）和DPI（点每英寸），从而可以将文本的像素宽度转换为英寸。
# axes: matplotlib.axes.Axes 对象，代表图表中的一个坐标轴（通常是图表中的一个子图）。这个参数用于获取和调整x轴的显示范围
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    # 获取文本在当前渲染器上的边界框，用于后续计算文本宽度
    bb = t.get_window_extent(renderer=r)
    # 将文本的边界框宽度从像素转换为英寸，这里使用了当前图形的DPI（点每英寸）作为转换因子
    text_width_inches = bb.width / fig.dpi
    # 获取当前图形的宽度（英寸）
    current_fig_width = fig.get_figwidth()
    # 计算新的图形宽度，即在当前宽度基础上增加文本的宽度，以确保文本完全显示在图形内
    new_fig_width = current_fig_width + text_width_inches
    # 计算新旧图形宽度的比例，这个比例将用于调整x轴的显示范围，以确保图形比例不变
    propotion = new_fig_width / current_fig_width
    # get axis limit
    # 获取当前x轴的显示范围
    x_lim = axes.get_xlim()
    # 根据新旧图形宽度的比例，调整x轴的显示范围。这里简单地将原范围的右边界乘以比例，
    # 这样做可能会导致x轴显示的数据范围失真，特别是如果文本宽度相对于图形宽度较大时。
    # 更合理的做法可能是调整图形大小或文本位置，而不是直接调整x轴范围。
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 使用Matplotlib绘制绘图
"""


# 定义一个函数，用于绘制条形图，参数包括字典、类别数、窗口标题、图表标题、x轴标签、输出路径、是否显示图表、条形图颜色、真实正例字典
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    # 使用operator.itemgetter对字典按值进行降序排序，结果是一个键值对元组的列表
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    # # 将排序后的键值对列表解包成两个列表：键列表和值列表
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # # 如果提供了真实正例字典（true_p_bar），则执行特殊处理
    if true_p_bar != "":
        """
         特殊情况：
            -绿色->TP：真阳性（检测到物体并匹配地面真实值）
            -红色->FP：假阳性（检测到物体但与地面实况不匹配）
            -橙色->FN：假阴性（未检测到但存在于地面真实中的对象）
        """
        # 初始化FP和TP列表
        fp_sorted = []
        tp_sorted = []
        # 遍历排序后的键，计算每个键的FP和TP
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        # 绘制FP条形图
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        # 绘制TP条形图，left参数将TP条形图向左偏移，使其叠加在FP条形图上
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        # 添加图例
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        # 在条形图旁边添加数值
        fig = plt.gcf()  # gcf - get current figure # 获取当前图形
        axes = plt.gca()  # 获取当前坐标轴
        r = fig.canvas.get_renderer()  # 获取渲染器
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            # 绘制数值，先绘制整个字符串，然后覆盖绘制TP部分
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            # 如果是最大的条形图，则可能需要调整坐标轴
            # if i == (len(sorted_values)-1):这行代码的目的是检查当前迭代的索引i是否是列表sorted_values中的最后一个元素的索引。只想对图表中的最后一个（也是最大的）条形图执行特定的操作，比如调整坐标轴以确保所有内容都可见。
            # 这里减1的原因是，Python中的列表索引是从0开始的。因此，如果sorted_values列表有n个元素，那么最后一个元素的索引将是n-1。len(sorted_values)返回列表的长度（即元素的数量），但这是一个从1开始的计数，而列表索引是从0开始的。所以，为了得到最后一个元素的索引，您需要从长度中减去1。
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        # 如果没有提供true_p_bar，则直接绘制普通条形图
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         在条形图的侧面写上数字
        """
        # 在条形图旁边添加数值
        fig = plt.gcf()  # gcf - get current figure # # 获取当前活动的图形对象，并将其赋值给变量fig
        axes = plt.gca()  # 获取当前活动的坐标轴对象，并将其赋值给变量axes
        r = fig.canvas.get_renderer()  # 获取当前图形（fig）的画布（canvas）的渲染器（renderer）对象，并将其赋值给变量r
        # # 遍历排序后的值列表，同时获取索引和值
        for i, val in enumerate(sorted_values):
            # # 将值转换为字符串，并在前面添加一个空格以改善显示效果
            str_val = " " + str(val)  # add a space before # 添加空格以改善显示效果
            # # 如果值小于1，则将该值格式化为两位小数的字符串
            if val < 1.0:
                str_val = " {0:.2f}".format(val)  # 如果值小于1，则格式化为两位小数
            # 在条形图的旁边绘制文本标签，位置根据值和索引确定，颜色、垂直对齐和字体加粗也进行了设置
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # 如果是最大的条形图（即列表中的最后一个元素），则可能需要调整坐标轴
            # re-set axes to show number inside the figure
            # 如果是最大的条形图，则可能需要调整坐标轴
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)  # 调用adjust_axes函数尝试调整坐标轴
    # set window title
    # 设置窗口标题
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    # 在y轴上设置刻度标签为排序后的键列表，并设置字体大小
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)  # 在y轴上显示类别名作为刻度标签，并设置字体大小
    """
     相应地重新调整高度
    """
    # 重新设置图形的高度以适应内容
    # 获取当前图形的高度（英寸）
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    # 计算DPI（点每英寸），这是图形分辨率的一个度量
    dpi = fig.dpi
    # 计算图形所需的总高度（以点为单位），这里考虑了类别数、字体大小和额外间距
    # n_classes 是类别数，tick_font_size 是y轴刻度标签的字体大小，1.4 是为了添加一些额外的间距
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    # 将所需的总高度从点转换为英寸
    height_in = height_pt / dpi
    # compute the required figure height
    # 计算所需的图形高度，考虑到顶部和底部的边距（以图形高度的百分比表示）
    top_margin = 0.15  # in percentage of the figure height # 顶部边距占图形高度的百分比
    bottom_margin = 0.05  # in percentage of the figure height # 底部边距占图形高度的百分比
    # 计算无边距时的图形高度，然后除以(1 - 顶部边距百分比 - 底部边距百分比)得到所需的图形高度
    # 在计算所需图形高度时，使用1 - top_margin - bottom_margin的原因是为了将顶部和底部边距从总高度中排除出去。这里的逻辑是这样的：
    # height_in 是根据类别数、字体大小和额外间距计算出的图形内容所需的最小高度（不包括边距）。
    # top_margin 和 bottom_margin 分别代表了图形顶部和底部的边距占图形总高度的百分比。如果我们想要图形内容（不包括边距）填充图形的大部分空间，那么我们就需要从总高度中减去这些边距所占的空间。
    # 因此，1 - top_margin - bottom_margin 就表示了图形内容（即不包括顶部和底部边距的部分）应该占图形总高度的比例。
    # 将height_in（内容所需高度）除以这个比例，就可以得到包含顶部和底部边距在内的图形总高度figure_height。
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    # 如果计算出的所需图形高度大于当前图形高度，则设置新的图形高度
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    # 设置图表的标题
    plt.title(plot_title, fontsize=14)  # 使用提供的标题和字体大小设置图表的标题
    # set axis titles
    # plt.xlabel('classes')
    # 设置x轴的标题，这里覆盖了之前的plt.xlabel('classes')调用，使用提供的x_label和字体大小
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    # 调整子图参数，使之填充整个图像区域。这会自动调整子图参数，使之优化显示
    fig.tight_layout()
    # save the plot
    # 保存图表到指定的文件路径
    fig.savefig(output_path)
    # show image
    # 如果to_show为True，则显示图表
    if to_show:
        plt.show()
    # close the plot
    # 关闭图表，释放资源
    plt.close()


# 计算并可视化目标检测模型在给定数据集上的性能指标，包括平均精度（Average Precision, AP）、F1分数、召回率（Recall）和精确度（Precision）。它通过比较模型的检测结果（Detection Results, DR）与真实标注（Ground Truth, GT）来评估模型的性能。此外，如果指定了图像路径（IMG_PATH），它还可以生成一个动画，展示每个检测结果的匹配情况
def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out'):
    GT_PATH = os.path.join(path, 'ground-truth')  # 设置路径变量
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')
    # 检查是否显示动画
    show_animation = True
    # 准备临时文件和结果文件目录
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False
    # 创建绘图所需的目录
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        os.makedirs(RESULTS_FILES_PATH)
    # 创建绘图所需的目录
    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))
    # 读取并处理真实标注文件
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # 初始化类别计数器和图像计数器
    gt_counter_per_class = {}
    counter_images_per_class = {}
    # 遍历真实标注文件，解析边界框并保存到临时文件
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:  # 遍历文件中的每一行
            try:
                # 尝试直接按空格分割每行，期望格式是 "class_name left top right bottom [difficult]" 或 "class_name left top right bottom"
                if "difficult" in line:  # 如果行中包含"difficult"标记
                    class_name, left, top, right, bottom, _difficult = line.split()  # 行格式正确，直接分割并赋值
                    is_difficult = True  # 标记为困难样本
                else:
                    class_name, left, top, right, bottom = line.split()  # 不含"difficult"标记，直接分割并赋值
            # 如果上述尝试失败，则进入异常处理
            except:
                if "difficult" in line:  # 如果行中仍然包含"difficult"标记
                    line_split = line.split()  # 再次尝试按空格分割整行
                    # 从后往前取值，因为格式可能不正确，所以直接从字符串列表中取最后几个元素
                    _difficult = line_split[-1]  # 最后一个元素视为"difficult"标记（尽管可能不是）
                    bottom = line_split[-2]  # 倒数第二个元素视为bottom坐标
                    right = line_split[-3]  # 倒数第三个元素视为right坐标
                    top = line_split[-4]  # 倒数第四个元素视为top坐标
                    left = line_split[-5]  # 倒数第五个元素视为left坐标
                    # 由于class_name可能由多个单词组成，因此需要从剩余部分拼接
                    class_name = ""
                    for name in line_split[:-5]:  # 遍历除了最后五个元素之外的所有元素
                        class_name += name + " "  # 拼接类名，注意末尾会多一个空格
                    class_name = class_name[:-1]  # 去除末尾多余的空格
                    is_difficult = True  # 标记为困难样本
                else:
                    # 如果行中不包含"difficult"标记且格式仍然不正确，则进行类似处理但不考虑"difficult"
                    line_split = line.split()
                    bottom = line_split[-1]  # 最后一个元素视为bottom坐标
                    right = line_split[-2]  # 倒数第二个元素视为right坐标
                    top = line_split[-3]  # 倒数第三个元素视为top坐标
                    left = line_split[-4]  # 倒数第四个元素视为left坐标
                    class_name = ""
                    for name in line_split[:-4]:  # 遍历除了最后四个元素之外的所有元素
                        class_name += name + " "  # 拼接类名
                    class_name = class_name[:-1]  # 去除末尾多余的空格
                # 构造边界框字符串，格式为 "left top right bottom"
            bbox = left + " " + top + " " + right + " " + bottom
            # 如果当前边界框被标记为困难样本
            if is_difficult:
                # 将边界框信息、类别名、使用标记（未使用）和困难标记添加到边界框列表中
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                # 重置困难标记（尽管这里重置可能是多余的，因为在下一次迭代中它会被重新赋值）
                is_difficult = False
            else:
                # 对于非困难样本，同样将边界框信息、类别名和使用标记添加到边界框列表中
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # 如果当前类别已经在类别计数器中存在，则增加其计数
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # 否则，在类别计数器中为该类别添加条目并初始化为1
                    gt_counter_per_class[class_name] = 1
                # 如果当前类别尚未记录在已见类别列表中
                if class_name not in already_seen_classes:
                    # 如果该类别已在图像计数器中存在，则增加其计数
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # 否则，在图像计数器中为该类别添加条目并初始化为1
                        counter_images_per_class[class_name] = 1
                    # 将当前类别添加到已见类别列表中
                    already_seen_classes.append(class_name)
            # 将解析后的边界框列表保存到临时JSON文件中
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    # 从类别计数器中获取所有类别，并对其进行排序
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)  # gt_counter_per_class中提取所有的键（keys），然后将这些键排序，并将排序后的结果赋值给变量gt_classes
    # 计算类别总数
    n_classes = len(gt_classes)

    # 查找并排序检测结果文件列表
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    # 遍历每个类别
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []  # 为当前类别初始化边界框列表
        for txt_file in dr_files_list:
            # 提取文件名（不包括扩展名）
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # 构造一个临时路径（这里可能是个错误，因为通常不需要再次访问GT_PATH）
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            # 对于第一个类别，检查对应的真实标注文件是否存在（这里的逻辑可能是不必要的，因为已经排序）
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)  # 注意：这里假设有一个名为'error'的函数来处理错误
            # 读取当前检测结果文件的行
            lines = file_lines_to_list(txt_file)
            # 遍历文件中的每一行
            for line in lines:
                try:
                    # 尝试按空格分割行并提取所需信息
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    # 如果分割失败，则尝试另一种解析方式
                    line_split = line.split()  # 再次尝试按空格分割当前行，但这次不指定分割次数
                    # 从分割后的列表中逆序提取边界框的坐标和置信度
                    bottom = line_split[-1]  # 提取最后一个元素作为底部坐标
                    right = line_split[-2]  # 提取倒数第二个元素作为右侧坐标
                    top = line_split[-3]  # 提取倒数第三个元素作为顶部坐标
                    left = line_split[-4]  # 提取倒数第四个元素作为左侧坐标
                    confidence = line_split[-5]  # 提取倒数第五个元素作为置信度
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]
                # 如果当前行的类别与正在处理的类别匹配
                if tmp_class_name == class_name:
                    # 构造边界框字符串并添加到边界框列表中
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # 按置信度降序排序当前类别的边界框列表
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        # 将排序后的边界框列表保存到临时JSON文件中
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    # 初始化平均精度（AP）总和为0.0
    sum_AP = 0.0
    # 初始化一个字典来存储每个类别的AP值
    ap_dictionary = {}
    # 初始化一个字典来存储每个类别的对数平均漏检率（LAMR）值（尽管在提供的代码段中LAMR未被计算）
    lamr_dictionary = {}
    # 打开结果文件并准备写入
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        # 写入结果文件的标题
        results_file.write("# AP and precision/recall per class\n")
        # 初始化一个字典来计数每个类别的真正例（True Positives, TP）
        count_true_positives = {}

        # 遍历所有真实标注中出现的类别
        for class_index, class_name in enumerate(gt_classes):
            # 初始化当前类别的真正例计数为0
            count_true_positives[class_name] = 0
            # 构造当前类别的检测结果文件路径
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            # 加载检测结果数据
            dr_data = json.load(open(dr_file))

            # 获取检测结果的数量
            nd = len(dr_data)
            # 初始化真正例（TP）、假正例（FP）和分数列表，长度与检测结果数量相同
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            # 初始化分数阈值索引（用于后续找到满足阈值的最高排名检测结果）
            score_threhold_idx = 0
            # 遍历所有检测结果
            for idx, detection in enumerate(dr_data):
                # 提取当前检测结果的文件ID和置信度分数
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                # 如果当前检测结果的置信度分数高于或等于设定的分数阈值
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx  # 更新分数阈值索引

                # 如果启用了动画显示
                if show_animation:
                    # 尝试根据文件ID找到对应的真实标注图像
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    # 如果没有找到图像，则报错
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    # 如果找到了多个图像，也报错（尽管这种情况在实际应用中不太可能）
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:
                        # 读取图像文件
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[
                            0])  # 通过索引 [0] 访问了 ground_truth_img 列表的第一个元素，即与当前处理结果相关联的第一个（或唯一找到的）图像文件的名称
                        # 构造累积图像的路径（尽管在提供的代码段中未使用）
                        img_cumulative_path = RESULTS_FILES_PATH + "/images/" + ground_truth_img[0]
                        # 尝试读取累积图像（如果已存在）或复制原始图像
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # 为图像添加底部边框（尽管在提供的代码段中未进一步使用修改后的图像）
                        bottom_border = 60
                        # 定义黑色边框的颜色值，这里使用的是BGR格式（OpenCV默认的颜色格式）
                        BLACK = [0, 0, 0]
                        # 给图像添加底部边框，边框颜色为黑色，边框高度为bottom_border，边框位于图像底部
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    # 构造真实标注文件的路径
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                # 加载真实标注数据
                ground_truth_data = json.load(open(gt_file))
                # 初始化最大交并比（IoU）为-1，用于后续比较
                ovmax = -1
                # 初始化匹配的真实标注对象为-1，表示尚未找到匹配
                gt_match = -1
                # 解析当前检测结果的边界框坐标，并转换为浮点数列表
                bb = [float(x) for x in detection["bbox"].split()]
                # 遍历真实标注数据中的每一个对象
                for obj in ground_truth_data:
                    # 如果当前真实标注对象的类别与当前处理的类别相同
                    if obj["class_name"] == class_name:
                        # 解析真实标注对象的边界框坐标，并转换为浮点数列表
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # 计算检测结果边界框与真实标注边界框的交集区域
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        # 用于计算两个边界框（bounding boxes）交集区域的宽度（iw）和高度（ih）。
                        # 在计算交集区域的尺寸时，我们需要考虑的是交集区域的“内部”尺寸，而不是边界框本身的尺寸。具体来说：
                        # bi[0] 和 bi[2] 分别是交集区域左上角和右下角的 x 坐标。
                        # bi[1] 和 bi[3] 分别是交集区域左上角和右下角的 y 坐标。
                        # 计算交集区域的宽度时，我们需要找到交集在 x 轴方向上的跨度。这个跨度是从交集区域的左边界（bi[0]）到右边界（bi[2]）的距离。但是，由于 x_max 和 x_min 是边界框的坐标，它们之间实际上包括了边界框的边界线。因此，为了得到交集区域的“内部”宽度，我们需要从 bi[2]（右边界的 x 坐标）中减去 bi[0]（左边界的 x 坐标），然后再加 1。这个加 1 的操作是为了将原本由整数坐标定义的边界框宽度（它实际上是一个“开区间”的右端点减去左端点，不包含右端点）转换为一个包含整个像素宽度的“闭区间”尺寸。
                        # 同样的逻辑也适用于高度计算（ih）。
                        # 计算交集区域的宽度和高度
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        # 如果交集区域存在（即宽度和高度都大于0）
                        if iw > 0 and ih > 0:
                            # 计算并集区域的面积
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            # 计算交并比（IoU）
                            ov = iw * ih / ua
                            # 如果当前计算出的交并比大于之前记录的最大交并比
                            if ov > ovmax:
                                ovmax = ov  # 更新最大交并比
                                gt_match = obj  # 更新匹配的真实标注对象
                            # 如果启用了动画显示
                if show_animation:
                    status = "NO MATCH FOUND!"  # 初始化匹配状态为未找到匹配
                # 定义最小交并比阈值
                min_overlap = MINOVERLAP
                # 如果最大交并比大于或等于最小交并比阈值
                if ovmax >= min_overlap:
                    # 如果匹配的真实标注对象没有被标记为“difficult”且未被使用过
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # 将当前检测结果标记为真正例（TP）
                            tp[idx] = 1  # 1 表示“是”或“满足条件”，而 0表示“否”或“不满足条件”
                            # 标记匹配的真实标注对象为已使用
                            gt_match["used"] = True
                            # 更新该类别的真正例计数
                            count_true_positives[class_name] += 1
                            # 更新真实标注文件，记录匹配状态
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            # 如果启用了动画显示，则更新匹配状态为找到匹配
                            if show_animation:
                                status = "MATCH!"
                        else:
                            # 如果匹配的真实标注对象已被使用过，则当前检测结果为假正例（FP）
                            fp[idx] = 1
                            # 如果启用了动画显示，则更新匹配状态为重复匹配（尽管在实际情况中这通常不是有效的匹配状态）
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    # 如果最大交并比小于最小交并比阈值，则当前检测结果为假正例（FP）
                    fp[idx] = 1
                    # 如果最大交并比大于0但小于阈值，更新匹配状态为交集区域不足
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                绘制图像以显示动画
                """
                # 如果启用了动画显示
                if show_animation:
                    # 获取图像的高度和宽度
                    height, widht = img.shape[:2]
                    # 定义要在图像上绘制的文本颜色
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    margin = 10  # 定义图像边缘的空白边距
                    # 1nd line
                    v_pos = int(height - margin - (bottom_border / 2.0))  # 第一行文本的位置计算
                    text = "Image: " + ground_truth_img[0] + " "
                    # 在图像上绘制第一行文本，并获取绘制后的行宽
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                    # 在图像上继续绘制第二行文本（实际上是第一行之后的文本），考虑前一行文本的宽度
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                         line_width)
                    if ovmax != -1:  # 如果找到了交集区域
                        color = light_red
                        # 根据匹配状态设置文本内容和颜色
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        # 在图像上绘制IoU信息
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 2nd line
                    # 第二行文本的位置计算
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx + 1)  # 检测结果的排名（从1开始计数）
                    text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        float(detection["confidence"]) * 100)
                    # 在图像上绘制第二行文本
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "MATCH!":
                        color = green  # 如果匹配成功，则改变文本颜色
                    text = "Result: " + status + " "
                    # 在图像上继续绘制关于匹配状态的信息
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 绘制边界框和类别名
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        # 在原始图像和累积图像上绘制真实标注的边界框
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    # 在原始图像和累积图像上绘制检测结果的边界框
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                    # 显示图像
                    cv2.imshow("Animation", img)
                    cv2.waitKey(20)  # 等待20毫秒后继续执行
                    # 保存处理后的图像
                    output_img_path = RESULTS_FILES_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                        idx) + ".jpg"
                    cv2.imwrite(output_img_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                               (np.array(prec) + np.array(rec)))

            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec) > 0:
                F1_text = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                Recall_text = "{0:.2f}%".format(rec[score_threhold_idx] * 100) + " = " + class_name + " Recall "
                Precision_text = "{0:.2f}%".format(prec[score_threhold_idx] * 100) + " = " + class_name + " Precision "
            else:
                F1_text = "0.00" + " = " + class_name + " F1 "
                Recall_text = "0.00%" + " = " + class_name + " Recall "
                Precision_text = "0.00%" + " = " + class_name + " Precision "

            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if len(prec) > 0:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=" + "{0:.2f}".format(
                    F1[score_threhold_idx]) \
                      + " ; Recall=" + "{0:.2f}%".format(
                    rec[score_threhold_idx] * 100) + " ; Precision=" + "{0:.2f}%".format(
                    prec[score_threhold_idx] * 100))
            else:
                print(text + "\t||\tscore_threhold=" + str(
                    score_threhold) + " : " + "F1=0.00% ; Recall=0.00% ; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                fig = plt.gcf()
                fig.canvas.set_window_title('AP ' + class_name)

                plt.title('class: ' + text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/AP/" + class_name + ".png")
                plt.cla()

                plt.plot(score, F1, "-", color='orangered')
                plt.title('class: ' + F1_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('F1')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/F1/" + class_name + ".png")
                plt.cla()

                plt.plot(score, rec, "-H", color='gold')
                plt.title('class: ' + Recall_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Recall')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Recall/" + class_name + ".png")
                plt.cla()

                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title('class: ' + Precision_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Precision/" + class_name + ".png")
                plt.cla()
        # 如果启用了动画显示，则关闭所有OpenCV窗口
        if show_animation:
            cv2.destroyAllWindows()
        # 检查是否检测到了任何类别，如果没有，则打印错误信息并返回0
        if n_classes == 0:
            print("未检测到任何种类，请检查标签信息与get_map.py中的classes_path是否修改。")
            return 0
        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(
        TEMP_FILES_PATH)  # shutil.rmtree(path) 方法在Python中用于递归地删除一个目录树。这个方法会删除指定的目录及其包含的所有子目录和文件 path：要删除的目录的路径。这个路径可以是绝对路径也可以是相对路径

    """
    统计检测结果总数
    """
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    """
    将每个类的地面真实对象数写入results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
    完成真阳性计数
    """
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    """
    将每个类检测到的对象数写入results.txt
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    """
    绘制每个类在地面实况中的总出现次数
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = RESULTS_FILES_PATH + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    # """
    # Plot the total number of occurences of each class in the "detection-results" folder
    # """
    # if draw_plot:
    #     window_title = "detection-results-info"
    #     # Plot title
    #     plot_title = "detection-results\n"
    #     plot_title += "(" + str(len(dr_files_list)) + " files and "
    #     count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    #     plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    #     # end Plot title
    #     x_label = "Number of objects per class"
    #     output_path = RESULTS_FILES_PATH + "/detection-results-info.png"
    #     to_show = False
    #     plot_color = 'forestgreen'
    #     true_p_bar = count_true_positives
    #     draw_plot_func(
    #         det_counter_per_class,
    #         len(det_counter_per_class),
    #         window_title,
    #         plot_title,
    #         x_label,
    #         output_path,
    #         to_show,
    #         plot_color,
    #         true_p_bar
    #         )

    """
    绘制对数平均漏报率图（按降序显示所有班级的漏报率）
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = RESULTS_FILES_PATH + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )

    """
    绘制mAP图（按降序显示所有类别的AP）
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = RESULTS_FILES_PATH + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )
    return mAP


def preprocess_gt(gt_path, class_names):
    image_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        # -----------------------------------------------------------------#
        #
        #   解决'Results do not correspond to current coco set'问题
        # -----------------------------------------------------------------#
        image['id'] = str(image_id)

        for line in lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]

            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id,
                    (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images'] = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories.append(category)
    results['categories'] = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results


def preprocess_dr(dr_path, class_names):
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result = {}
            result["image_id"] = str(image_id)
            if class_name not in class_names:
                continue
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left, top, right - left, bottom - top]
            result["score"] = float(confidence)
            results.append(result)
    return results


def get_coco_map(class_names, path):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)
        if len(results_dr) == 0:
            print("未检测到任何目标。")
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cocoGt = COCO(GT_JSON_PATH)
    cocoDt = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats