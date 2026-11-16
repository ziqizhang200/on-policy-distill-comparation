"""Generate the two grayscale method diagrams embedded in the opening report."""

from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
INK = '#22272b'
LINE = '#6c7378'
FILL = '#f1f2f2'


def label(ax, x, y, value, size=14, weight='normal', align='center'):
    ax.text(x, y, value, ha=align, va='center', color=INK,
            fontsize=size, fontproperties=FONT, weight=weight, linespacing=1.5)


def box(ax, x, y, w, h, text, size=14, fill='white'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=LINE, lw=1.15))
    label(ax, x + w / 2, y + h / 2, text, size)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                                 lw=1.0, color=LINE, shrinkA=0, shrinkB=0))


def setup(width, height):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def figure_one():
    fig, ax = setup(1351, 608)
    labels = [('研究问题', .73), ('研究方法', .44), ('研究目标', .15)]
    cols = [.105, .405, .705]
    width = .245
    height = .19
    rows = [
        ['关键决策位置怎样\n匹配更新目标？', '有限反馈怎样保留\n关键候选与尾部质量？', '动态学生轨迹怎样\n进行可信比较？'],
        ['决策状态选择与\n双向散度联合调节', '决策感知的\n稀疏支持集', '配对回放与闭环训练的\n统一测试系统'],
        ['提高关键位置的\n有效监督', '保留有限预算下的\n决策信号', '追踪局部机制、\n调用错误与成本'],
    ]
    for rid, (name, y) in enumerate(labels):
        box(ax, .012, y, .075, height, name, 14, FILL)
        for cid, x in enumerate(cols):
            box(ax, x, y, width, height, rows[rid][cid], 14,
                FILL if rid == 1 else 'white')
            if cid < 2:
                arrow(ax, (x + width + .008, y + height / 2),
                      (cols[cid + 1] - .008, y + height / 2))
            if rid < 2:
                arrow(ax, (x + width / 2, y - .012),
                      (x + width / 2, labels[rid + 1][1] + height + .012))
    label(ax, .5, .045, '研究一：位置与目标联合设计  →  研究二：有限反馈改进  →  研究三：系统化验证', 12)
    fig.savefig(ROOT / 'figure1_method_levels.png', dpi=100)
    plt.close(fig)


def figure_two():
    fig, ax = setup(1397, 524)
    xvals = [.035, .23, .425, .62, .815]
    w, h, y = .15, .45, .35
    items = [
        '输入与轨迹\n单轮工具调用\n教师同前缀评分',
        '研究一\n状态与散度联合\n确定位置和方向',
        '研究二\n决策感知稀疏反馈\n保留候选和尾部质量',
        '研究三\n统一测试系统\n配对回放与闭环训练',
        '研究产出\n调用表现与成本\n可追溯证据链',
    ]
    for i, x in enumerate(xvals):
        box(ax, x, y, w, h, items[i], 13.5, FILL if 1 <= i <= 3 else 'white')
        if i < 4:
            arrow(ax, (x + w + .008, y + h / 2),
                  (xvals[i + 1] - .008, y + h / 2))
    label(ax, .5, .18, '完整词表检验联合方法；固定规则比较稀疏反馈；双路径验证机制与任务结果', 12)
    fig.savefig(ROOT / 'figure2_method_route.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    figure_one()
    figure_two()
