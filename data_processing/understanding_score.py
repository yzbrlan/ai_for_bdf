import matplotlib.pyplot as plt
import numpy as np

# 定义模型名称
models = ['deepseek-V3', 'Qwen2.5-72B', 'Qwen2.5-32B', 'Qwen2.5-14B', 'Qwen2.5-7B']

# 假设的模型大小（单位：GB）
model_sizes = [404, 47, 20, 9, 4.7]

# 假设的理解 BDF 指标上的分数
bdf_scores = [12, 13, 13, 13, 9]


def first_plot():
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300

    # 设置柱子的宽度
    width = 0.2

    # 生成 x 轴位置
    x = np.arange(len(models))

    # 创建一个图形和坐标轴
    fig, ax = plt.subplots()

    # 绘制模型大小的柱子
    rects1 = ax.bar(x - width/2, model_sizes, width, label='Model Size (GB)')

    # 绘制理解 BDF 指标分数的柱子
    rects2 = ax.bar(x + width/2, bdf_scores, width, label='BDF Understanding Score')

    # 设置 x 轴标签、标题和图例
    ax.set_ylabel('Values')
    ax.set_title('Model Size and BDF Understanding Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    # 添加数据标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # 显示图形
    plt.tight_layout()
    plt.savefig('understanding_score.png')

# plt.show()
def double_y_plot():
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    x = np.arange(len(models))

    fig, ax1 = plt.subplots()

    # 绘制模型大小
    color = 'tab:red'
    ax1.set_xlabel('Models', fontsize=8)
    ax1.set_ylabel('Model Size (GB)', color=color, fontsize=8)
    bars = ax1.bar(x - 0.2, model_sizes, 0.4, color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6)

    # 创建第二个 Y 轴
    ax2 = ax1.twinx()

    # 绘制 BDF 理解分数
    color = 'tab:blue'
    ax2.set_ylabel('BDF Understanding Score', color=color, fontsize=8)
    bars = ax2.bar(x + 0.2, bdf_scores, 0.4, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=8)
    ax2.tick_params(axis='x', labelsize=8)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6)

    # 设置 X 轴标签
    plt.xticks(x, models, rotation=45)

    # 设置标题
    plt.title('Model Size and BDF Understanding Score', fontsize=8)

    # 保存图片
    plt.savefig('understanding_score_double_y.png')
    plt.show()

def radar_plot():
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 数据
    normalized_sizes = np.array(model_sizes) / max(model_sizes)
    normalized_scores = np.array(bdf_scores) / max(bdf_scores)

    # 指标名称
    categories = ['Model Size', 'BDF Score']
    N = len(categories)

    # 为每个模型绘制雷达图
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.rcParams['figure.dpi'] = 300

    fig = plt.figure()
    for i, model in enumerate(models):
        values = [normalized_sizes[i], normalized_scores[i]]
        values += values[:1]
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)

    # 设置角度和标签
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # 保存图片
    # plt.savefig('understanding_score_radar.png')
    plt.show()

double_y_plot()