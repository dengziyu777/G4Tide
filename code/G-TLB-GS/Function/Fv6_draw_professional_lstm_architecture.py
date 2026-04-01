import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import os
import math
import joblib
import warnings
from matplotlib.font_manager import FontProperties


def Fv6_draw_professional_lstm_architecture(output_prediction_folder, input_size, hidden_sizes, output_size,
                                        bidirectional=False, dropout=0.0, use_forward_hours=24,
                                        time_step=1, figsize=(16, 12)):
    """
    Draw a professional and aesthetically pleasing LSTM neural network architecture diagram.
    """
    try:
        # Set output path
        os.makedirs(output_prediction_folder, exist_ok=True)
        output_path = os.path.join(output_prediction_folder, 'lstm_architecture.png')

        # Create figure and grid
        fig = plt.figure(figsize=figsize, facecolor='#F8F9FA')
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])

        # Set global styles
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold'
        })

        # Title
        title = f'LSTM Neural Network Architecture\nInput Size: {input_size} | Hidden Layers: {hidden_sizes} | Output Size: {output_size}'
        if bidirectional:
            title += " | Bidirectional"
        if dropout > 0:
            title += f" | Dropout: {dropout}"

        fig.suptitle(title, fontsize=18, fontweight='bold', color='#2C3E50')

        # Set coordinate system
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.axis('off')

        # Color scheme (professional dark tones)
        colors = {
            'input': '#3498DB',  # Blue
            'lstm': '#E74C3C',  # Red
            'lstm_rev': '#9B59B6',  # Purple
            'ln': '#F1C40F',  # Yellow
            'fc': '#2ECC71',  # Green
            'dropout': '#9B59B6',  # Purple
            'output': '#E67E22',  # Orange
            'text_dark': '#2C3E50',
            'text_light': '#ECF0F1'
        }

        # Calculate dynamic parameters
        time_steps = min(6, use_forward_hours // max(1, time_step))
        layer_spacing = 0.12
        block_height = 0.07
        block_width = 0.15
        x_pos = 0.1
        y_pos = 0.75

        # ==================== 1. Draw Input Sequence ====================
        input_y = y_pos
        input_block_height = block_height * min(4, input_size) / 1.5

        # Input sequence main area
        input_rect = Rectangle(
            (x_pos, input_y),
            block_width,
            input_block_height,
            fc=colors['input'],
            ec='#2C3E50',
            alpha=0.9,
            lw=1.5,
            zorder=10
        )
        ax.add_patch(input_rect)

        # Add text (as in the image example)
        plt.text(
            x_pos + block_width / 2,
            input_y + input_block_height / 2,
            f"Input Sequence\n{time_steps}×{input_size} features",
            ha='center',
            va='center',
            fontsize=12,
            color='white',
            fontweight='bold',
            zorder=11
        )

        # Time step indicators
        time_y = input_y + input_block_height + 0.02
        for t in range(time_steps):
            t_x = x_pos + (t / (time_steps - 1)) * block_width
            plt.plot([t_x, t_x], [time_y, time_y + 0.03], color=colors['input'], lw=2.5)
            plt.text(t_x, time_y + 0.04, f"t-{time_steps - t - 1}",
                     ha='center', fontsize=10, color=colors['text_dark'])

        # ==================== 2. Draw LSTM Layers ====================
        y_pos = input_y - layer_spacing - input_block_height
        layer_heights = []

        for layer_idx, hidden_size in enumerate(hidden_sizes):
            # Calculate position
            y_layer = y_pos - layer_idx * (block_height + layer_spacing)
            layer_heights.append(y_layer)

            # Main LSTM layer
            lstm_rect = Rectangle(
                (x_pos + 0.07, y_layer),
                block_width - 0.14,
                block_height,
                fc=colors['lstm'],
                ec='#2C3E50',
                alpha=0.9,
                lw=1.5,
                zorder=9
            )
            ax.add_patch(lstm_rect)

            # Add text
            plt.text(
                x_pos + block_width / 2,
                y_layer + block_height / 2,
                f"LSTM Layer {layer_idx + 1}\nSize: {hidden_size}",
                ha='center',
                va='center',
                fontsize=12,
                color='white',
                fontweight='bold',
                zorder=10
            )

            # Add connection arrows
            if layer_idx == 0:
                # From input layer to first LSTM layer
                arrow = FancyArrowPatch(
                    (x_pos + block_width / 2, input_y),
                    (x_pos + block_width / 2, y_layer + block_height),
                    arrowstyle='->,head_width=0.03,head_length=0.05',
                    mutation_scale=10,
                    lw=2,
                    color=colors['text_dark'],
                    alpha=0.8,
                    zorder=8
                )
                ax.add_patch(arrow)
            else:
                # From previous layer to current layer
                arrow = FancyArrowPatch(
                    (x_pos + block_width / 2, layer_heights[layer_idx - 1]),
                    (x_pos + block_width / 2, y_layer + block_height),
                    arrowstyle='->,head_width=0.03,head_length=0.05',
                    mutation_scale=10,
                    lw=2,
                    color=colors['text_dark'],
                    alpha=0.8,
                    zorder=8
                )
                ax.add_patch(arrow)

            # For bidirectional LSTM, add reverse layer
            if bidirectional:
                rev_y = y_layer - (block_height + layer_spacing * 0.6)
                rev_rect = Rectangle(
                    (x_pos + 0.07, rev_y),
                    block_width - 0.14,
                    block_height,
                    fc=colors['lstm_rev'],
                    ec='#2C3E50',
                    alpha=0.9,
                    lw=1.5,
                    zorder=9
                )
                ax.add_patch(rev_rect)

                plt.text(
                    x_pos + block_width / 2,
                    rev_y + block_height / 2,
                    f"Reverse LSTM {layer_idx + 1}",
                    ha='center',
                    va='center',
                    fontsize=12,
                    color='white',
                    fontweight='bold',
                    zorder=10
                )

                # Add connection from main layer to reverse layer
                arrow_rev = FancyArrowPatch(
                    (x_pos + block_width / 2, y_layer),
                    (x_pos + block_width / 2, rev_y + block_height),
                    arrowstyle='->,head_width=0.03,head_length=0.05',
                    mutation_scale=10,
                    lw=2,
                    color=colors['text_dark'],
                    alpha=0.6,
                    zorder=8
                )
                ax.add_patch(arrow_rev)

                # Update position
                y_pos = rev_y

            # Add Dropout layer (not for the last layer)
            if dropout > 0 and layer_idx < len(hidden_sizes) - 1:
                drop_y = y_layer - (block_height + layer_spacing * 0.3)
                drop_rect = Rectangle(
                    (x_pos + block_width / 4, drop_y),
                    block_width / 2,
                    block_height / 1.5,
                    fc=colors['dropout'],
                    ec='#2C3E50',
                    alpha=0.8,
                    lw=1.5,
                    zorder=9
                )
                ax.add_patch(drop_rect)

                plt.text(
                    x_pos + block_width / 2,
                    drop_y + block_height / 3,
                    f"Dropout: {dropout}",
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='white',
                    fontweight='bold',
                    zorder=10
                )

                # Add connection arrow to Dropout layer
                arrow_drop = FancyArrowPatch(
                    (x_pos + block_width / 2, y_layer),
                    (x_pos + block_width / 2, drop_y + block_height / 1.5),
                    arrowstyle='->,head_width=0.02,head_length=0.04',
                    mutation_scale=8,
                    lw=1.8,
                    color=colors['text_dark'],
                    alpha=0.7,
                    zorder=8
                )
                ax.add_patch(arrow_drop)

        # ==================== 3. Draw Layer Normalization ====================
        last_y = layer_heights[-1] if not bidirectional else y_pos
        ln_y = last_y - (block_height + layer_spacing * 0.8)

        ln_rect = Rectangle(
            (x_pos, ln_y),
            block_width,
            block_height,
            fc=colors['ln'],
            ec='#2C3E50',
            alpha=0.9,
            lw=1.5,
            zorder=9
        )
        ax.add_patch(ln_rect)

        plt.text(
            x_pos + block_width / 2,
            ln_y + block_height / 2,
            "Layer Normalization",
            ha='center',
            va='center',
            fontsize=12,
            color='white',
            fontweight='bold',
            zorder=10
        )

        # Add connection arrow from previous layer
        arrow_ln = FancyArrowPatch(
            (x_pos + block_width / 2, last_y),
            (x_pos + block_width / 2, ln_y + block_height),
            arrowstyle='->,head_width=0.03,head_length=0.05',
            mutation_scale=10,
            lw=2,
            color=colors['text_dark'],
            alpha=0.8,
            zorder=8
        )
        ax.add_patch(arrow_ln)

        # ==================== 4. Draw Fully Connected Layer ====================
        fc_y = ln_y - (block_height + layer_spacing)

        fc_rect = Rectangle(
            (x_pos, fc_y),
            block_width,
            block_height,
            fc=colors['fc'],
            ec='#2C3E50',
            alpha=0.9,
            lw=1.5,
            zorder=9
        )
        ax.add_patch(fc_rect)

        fc_size = hidden_sizes[-1] * (2 if bidirectional else 1) // 2
        plt.text(
            x_pos + block_width / 2,
            fc_y + block_height / 2,
            f"Fully Connected\nSize: {fc_size}",
            ha='center',
            va='center',
            fontsize=12,
            color='white',
            fontweight='bold',
            zorder=10
        )

        # Add connection arrow from previous layer
        arrow_fc = FancyArrowPatch(
            (x_pos + block_width / 2, ln_y),
            (x_pos + block_width / 2, fc_y + block_height),
            arrowstyle='->,head_width=0.03,head_length=0.05',
            mutation_scale=10,
            lw=2,
            color=colors['text_dark'],
            alpha=0.8,
            zorder=8
        )
        ax.add_patch(arrow_fc)

        # ==================== 5. Draw Output Layer ====================
        output_y = fc_y - (block_height + layer_spacing)

        output_rect = Rectangle(
            (x_pos, output_y),
            block_width,
            block_height,
            fc=colors['output'],
            ec='#2C3E50',
            alpha=0.9,
            lw=1.5,
            zorder=9
        )
        ax.add_patch(output_rect)

        plt.text(
            x_pos + block_width / 2,
            output_y + block_height / 2,
            "Output Layer\nTide Level Prediction",
            ha='center',
            va='center',
            fontsize=12,
            color='white',
            fontweight='bold',
            zorder=10
        )

        # Add connection arrow from previous layer
        arrow_out = FancyArrowPatch(
            (x_pos + block_width / 2, fc_y),
            (x_pos + block_width / 2, output_y + block_height),
            arrowstyle='->,head_width=0.03,head_length=0.05',
            mutation_scale=10,
            lw=2,
            color=colors['text_dark'],
            alpha=0.8,
            zorder=8
        )
        ax.add_patch(arrow_out)

        # ==================== 6. Add Legend ====================
        legend_x = 0.75
        legend_y = 0.85
        legend_spacing = 0.08

        legend_elements = [
            ('Input Layer', colors['input']),
            ('LSTM Layer', colors['lstm']),
            ('Reverse LSTM', colors['lstm_rev']),
            ('Layer Norm', colors['ln']),
            ('Fully Connected', colors['fc']),
            ('Output Layer', colors['output'])
        ]

        if dropout > 0:
            legend_elements.append(('Dropout', colors['dropout']))

        for i, (label, color) in enumerate(legend_elements):
            rect = Rectangle(
                (legend_x, legend_y - i * legend_spacing),
                0.04,
                0.04,
                fc=color,
                ec='#2C3E50',
                alpha=0.9
            )
            ax.add_patch(rect)

            plt.text(
                legend_x + 0.05,
                legend_y - i * legend_spacing + 0.02,
                label,
                ha='left',
                va='center',
                fontsize=12,
                color=colors['text_dark']
            )

        # ==================== 7. Add Caption ====================
        plt.figtext(0.5, 0.02,
                   "Neural Network Architecture Diagram | Time Steps: {} hours | Created with matplotlib".format(use_forward_hours),
                   ha="center", fontsize=12, color=colors['text_dark'],
                   bbox={"facecolor": "#ECF0F1", "alpha": 0.7, "pad": 5})

        # Save high-quality image
        plt.tight_layout(pad=4.0)
        plt.subplots_adjust(top=0.92, bottom=0.08)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Professional LSTM architecture diagram saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to draw professional LSTM architecture diagram: {e}")
        return False