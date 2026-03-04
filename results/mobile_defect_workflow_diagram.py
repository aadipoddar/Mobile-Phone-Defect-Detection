import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ------------------------------------------------------------
# Figure setup (large canvas + transparent background)
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(26, 20))
fig.patch.set_alpha(0.0)
ax.set_xlim(0, 28)
ax.set_ylim(-8, 18)
ax.axis('off')


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------
color_data = '#E8F4F8'
color_proc = '#EAF7EA'
color_cnn = '#FFE5E5'
color_vit = '#F5E5FF'
color_hybrid = '#FFF3E0'
color_eval = '#FFF9D9'
color_text = '#2C3E50'
edge_color = '#34495E'


# ------------------------------------------------------------
# Box sizes
# ------------------------------------------------------------
BOX_WIDTH = 5.2
BOX_HEIGHT = 1.7


def create_box(ax, x, y, width, height, text, color, fontsize=18, fontweight='bold'):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle='round,pad=0.08',
        edgecolor=edge_color,
        facecolor=color,
        linewidth=2.8,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha='center',
        va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        color=color_text,
        wrap=True,
    )


def create_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle='->,head_width=0.45,head_length=0.35',
        color=edge_color,
        linewidth=3.2,
        connectionstyle='arc3,rad=0',
    )
    ax.add_patch(arrow)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
ax.text(
    14,
    17,
    'Mobile Phone Defect Classification - Research Workflow',
    ha='center',
    va='center',
    fontsize=46,
    fontweight='bold',
    color=color_text,
)


# ------------------------------------------------------------
# Left pipeline: data + preprocessing
# ------------------------------------------------------------
left_x = 1.2
create_box(ax, left_x, 13.2, BOX_WIDTH, BOX_HEIGHT, 'Dataset\n1,220 Images\n(4 Classes)', color_data, 28)
create_arrow(ax, left_x + BOX_WIDTH / 2, 13.2, left_x + BOX_WIDTH / 2, 10.7 + BOX_HEIGHT)

create_box(ax, left_x, 10.7, BOX_WIDTH, BOX_HEIGHT, 'Label Mapping\nGood / Oil / Scratch / Stain', color_data, 17)
create_arrow(ax, left_x + BOX_WIDTH / 2, 10.7, left_x + BOX_WIDTH / 2, 8.2 + BOX_HEIGHT)

create_box(ax, left_x, 8.2, BOX_WIDTH, BOX_HEIGHT, 'Split\nTrain / Val / Test\n70% / 15% / 15%', color_proc, 21)
create_arrow(ax, left_x + BOX_WIDTH / 2, 8.2, left_x + BOX_WIDTH / 2, 5.7 + BOX_HEIGHT)

create_box(ax, left_x, 5.7, BOX_WIDTH, BOX_HEIGHT, 'Preprocessing\nResize 224x224 + Normalize\nAugmentation (Train)', color_proc, 14)
create_arrow(ax, left_x + BOX_WIDTH / 2, 5.7, left_x + BOX_WIDTH / 2, 3.2 + BOX_HEIGHT)

create_box(ax, left_x, 3.2, BOX_WIDTH, BOX_HEIGHT, 'Feature Pipeline\nFrozen Backbones + FAISS Cache\nSMOTE + Weighted Sampling', color_proc, 13)


# ------------------------------------------------------------
# Branch to 3 tracks
# ------------------------------------------------------------
branch_y = 3.95
track1_x = 7.8   # CNN track
track2_x = 14.0  # Transformer track
track3_x = 20.2  # Hybrid track

create_arrow(ax, left_x + BOX_WIDTH, branch_y, track1_x + BOX_WIDTH / 2, branch_y)
create_arrow(ax, left_x + BOX_WIDTH, branch_y, track2_x + BOX_WIDTH / 2, branch_y)
create_arrow(ax, left_x + BOX_WIDTH, branch_y, track3_x + BOX_WIDTH / 2, branch_y)

create_arrow(ax, track1_x + BOX_WIDTH / 2, branch_y, track1_x + BOX_WIDTH / 2, 1.5 + BOX_HEIGHT)
create_arrow(ax, track2_x + BOX_WIDTH / 2, branch_y, track2_x + BOX_WIDTH / 2, 1.5 + BOX_HEIGHT)
create_arrow(ax, track3_x + BOX_WIDTH / 2, branch_y, track3_x + BOX_WIDTH / 2, 1.5 + BOX_HEIGHT)


# ------------------------------------------------------------
# Track 1: CNN models
# ------------------------------------------------------------
ax.text(
    track1_x + BOX_WIDTH / 2,
    5.2,
    'Track 1: CNN Models',
    ha='center',
    va='center',
    fontsize=24,
    fontweight='bold',
    color=color_text,
    bbox=dict(boxstyle='round,pad=0.35', facecolor=color_cnn, edgecolor=edge_color, linewidth=2.5),
)

create_box(ax, track1_x, 1.5, BOX_WIDTH, BOX_HEIGHT, 'VGG16', color_cnn, 30)
create_arrow(ax, track1_x + BOX_WIDTH / 2, 1.5, track1_x + BOX_WIDTH / 2, -1.0 + BOX_HEIGHT)

create_box(ax, track1_x, -1.0, BOX_WIDTH, BOX_HEIGHT, 'ResNet50', color_cnn, 28)
create_arrow(ax, track1_x + BOX_WIDTH / 2, -1.0, track1_x + BOX_WIDTH / 2, -3.5 + BOX_HEIGHT)

create_box(ax, track1_x, -3.5, BOX_WIDTH, BOX_HEIGHT, 'MobileNetV2\nEfficientNet-B0\nConvNeXt-Tiny', color_cnn, 18)


# ------------------------------------------------------------
# Track 2: Transformer models
# ------------------------------------------------------------
ax.text(
    track2_x + BOX_WIDTH / 2,
    5.2,
    'Track 2: Vision Transformers',
    ha='center',
    va='center',
    fontsize=24,
    fontweight='bold',
    color=color_text,
    bbox=dict(boxstyle='round,pad=0.35', facecolor=color_vit, edgecolor=edge_color, linewidth=2.5),
)

create_box(ax, track2_x, 1.5, BOX_WIDTH, BOX_HEIGHT, 'DeiT-Small', color_vit, 30)
create_arrow(ax, track2_x + BOX_WIDTH / 2, 1.5, track2_x + BOX_WIDTH / 2, -1.0 + BOX_HEIGHT)

create_box(ax, track2_x, -1.0, BOX_WIDTH, BOX_HEIGHT, 'Swin-Tiny', color_vit, 28)
create_arrow(ax, track2_x + BOX_WIDTH / 2, -1.0, track2_x + BOX_WIDTH / 2, -3.5 + BOX_HEIGHT)

create_box(ax, track2_x, -3.5, BOX_WIDTH, BOX_HEIGHT, 'DINOv2-Small', color_vit, 27)


# ------------------------------------------------------------
# Track 3: Hybrid models
# ------------------------------------------------------------
ax.text(
    track3_x + BOX_WIDTH / 2,
    5.2,
    'Track 3: Hybrid Models',
    ha='center',
    va='center',
    fontsize=24,
    fontweight='bold',
    color=color_text,
    bbox=dict(boxstyle='round,pad=0.35', facecolor=color_hybrid, edgecolor=edge_color, linewidth=2.5),
)

create_box(ax, track3_x, 1.5, BOX_WIDTH, BOX_HEIGHT, 'Model 9\nCNN Feature Fusion\n+ PCA + MLP', color_hybrid, 16)
create_arrow(ax, track3_x + BOX_WIDTH / 2, 1.5, track3_x + BOX_WIDTH / 2, -1.0 + BOX_HEIGHT)

create_box(ax, track3_x, -1.0, BOX_WIDTH, BOX_HEIGHT, 'Model 10\nDINOv2 + PCA\n+ XGBoost', color_hybrid, 16)
create_arrow(ax, track3_x + BOX_WIDTH / 2, -1.0, track3_x + BOX_WIDTH / 2, -3.5 + BOX_HEIGHT)

create_box(ax, track3_x, -3.5, BOX_WIDTH, BOX_HEIGHT, 'GridSearchCV\nHyperparameter Tuning', color_hybrid, 16)


# ------------------------------------------------------------
# Final evaluation block
# ------------------------------------------------------------
eval_x = 6.5
eval_y = -7.0
eval_w = 18.0
create_box(
    ax,
    eval_x,
    eval_y,
    eval_w,
    BOX_HEIGHT,
    'Evaluation: Accuracy, F1, AUC, Confusion Matrix, ROC\nBest Model: DeiT-Small (99.45% Accuracy)',
    color_eval,
    24,
)

# Arrows from all 3 tracks into evaluation
create_arrow(ax, track1_x + BOX_WIDTH / 2, -3.5, track1_x + BOX_WIDTH / 2, eval_y + BOX_HEIGHT)
create_arrow(ax, track2_x + BOX_WIDTH / 2, -3.5, track2_x + BOX_WIDTH / 2, eval_y + BOX_HEIGHT)
create_arrow(ax, track3_x + BOX_WIDTH / 2, -3.5, track3_x + BOX_WIDTH / 2, eval_y + BOX_HEIGHT)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------
legend_y = 15.5
legend_x_start = 3.8
legend_spacing = 4.25

legend_items = [
    ('Data', color_data),
    ('Preprocessing', color_proc),
    ('CNN', color_cnn),
    ('Transformer', color_vit),
    ('Hybrid / Results', color_hybrid),
]

for i, (label, color) in enumerate(legend_items):
    x_pos = legend_x_start + i * legend_spacing
    legend_box = mpatches.Rectangle(
        (x_pos, legend_y),
        0.55,
        0.42,
        facecolor=color,
        edgecolor=edge_color,
        linewidth=2,
    )
    ax.add_patch(legend_box)
    ax.text(
        x_pos + 0.75,
        legend_y + 0.21,
        label,
        ha='left',
        va='center',
        fontsize=22,
        fontweight='bold',
        color=color_text,
    )


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
plt.tight_layout()
output_file = 'Mobile_Defect_Model_Workflow.png'
plt.savefig(
    output_file,
    dpi=300,
    bbox_inches='tight',
    transparent=True,
    facecolor='none',
    edgecolor='none',
)

print('✅ Workflow diagram created successfully!')
print(f'   File: {output_file}')
print('   Size: 26x20 inches @ 300 DPI')
print('   Background: Transparent')
print('   Includes: Data -> FAISS/SMOTE/PCA -> 3 Model Tracks -> Final Evaluation')

plt.show()
