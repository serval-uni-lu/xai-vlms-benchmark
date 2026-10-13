import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from captum.attr import visualization
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch import Tensor


class XAIVisualizer:
    def __init__(self, processor):
        """
        Args:
            processor: The Hugging Face processor/tokenizer.
        """
        self.processor = processor

    def _get_cmap(self):
        cmap = LinearSegmentedColormap.from_list(
            "red_blue",
            [
                (0.0, "#0000ff"),  # blue   (low values)
                (0.5, "#ffffff"),  # white  (mid)
                (1.0, "#ff0000"),  # red    (high values)
            ],
            N=256,
        )
        return cmap

    def _resolve_target_ids(self, target_ids: Tensor, target_indices: int | list[int] | None) -> Tensor:
        """
        Helper to cleanly slice the target_ids tensor based on the requested indices.
        Returns a 1D tensor of the specific token IDs we are explaining.
        """
        flat_targets = target_ids[0]  # Assume shape is (1, seq_len)

        if target_indices is None:
            return flat_targets  # Return all of them

        indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Safely extract only the requested tokens
        valid_indices = [idx for idx in indices if idx < len(flat_targets)]
        return flat_targets[valid_indices]

    def plot_text_attributions(
        self,
        text_attr: Tensor,
        input_ids: Tensor,
        target_ids: Tensor,
        special_token_ids: list | None = None,
        semantic_mask: Tensor | None = None,
        normalize: bool = True,
        target_indices: int | list[int] | None = None,
    ):  # <--- ADDED
        """
        Visualizes text attributions, filtering out visual/special tokens.
        """
        special_token_ids = special_token_ids or []
        input_ids_list = input_ids[0].tolist()

        # 1. Identify valid prompt tokens (the text being highlighted)
        valid_indices = []
        for i, tok_id in enumerate(input_ids_list):
            if tok_id in special_token_ids:
                continue
            if semantic_mask is not None and not semantic_mask[i].item():
                continue
            valid_indices.append(i)

        if len(valid_indices) == 0:
            print("[!] Warning: No valid text tokens found to visualize.")
            return

        filtered_input_ids = input_ids[0][valid_indices]
        filtered_text_attr = text_attr[:, valid_indices]

        # 3. Decode prompt tokens
        tokens = self.processor.batch_decode(
            filtered_input_ids.unsqueeze(1),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(tokens[0], list):
            tokens = [t[0] for t in tokens]

        # --- NEW: Get the exact target tokens we are explaining ---
        specific_target_ids = self._resolve_target_ids(target_ids, target_indices)

        records = []
        num_answer_tokens = filtered_text_attr.shape[0]

        for i in range(num_answer_tokens):
            attr_scores = filtered_text_attr[i].cpu().detach().numpy()

            if normalize:
                max_abs = np.max(np.abs(attr_scores))
                if max_abs > 0:
                    attr_scores = attr_scores / max_abs

            # --- NEW: Decode the exact matched token ---
            target_token_str = self.processor.decode([specific_target_ids[i]])

            record = visualization.VisualizationDataRecord(
                word_attributions=attr_scores,
                pred_prob=1.0,
                pred_class=target_token_str,
                true_class=target_token_str,
                attr_class=target_token_str,
                attr_score=attr_scores.sum(),
                raw_input_ids=tokens,
                convergence_score=0.0,
            )
            records.append(record)

        print("\n" + "=" * 50)
        print("TEXT ATTRIBUTIONS (Filtered & Normalized)")
        print("=" * 50)
        _ = visualization.visualize_text(records)

    def plot_image_attributions(
        self,
        img_attr: Tensor,
        original_image: Image.Image,
        target_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        target_indices: int | list[int] | None = None,
    ):
        """
        Visualizes image attributions dynamically.
        """
        num_tokens = img_attr.shape[0]
        orig_w, orig_h = original_image.size

        # --- NEW: Get the exact target tokens we are explaining ---
        specific_target_ids = self._resolve_target_ids(target_ids, target_indices)

        # 1. Reshape Attributions
        if img_attr.dim() == 2:
            num_patches = img_attr.shape[1]
            if image_grid_thw is not None:
                grid_h = int(image_grid_thw[0, 1].item())
                grid_w = int(image_grid_thw[0, 2].item())
            else:
                grid_h = grid_w = int(np.sqrt(num_patches))
            attrs_2d = img_attr.view(num_tokens, grid_h, grid_w)

        elif img_attr.dim() == 3:
            attrs_2d = img_attr

        elif img_attr.dim() == 4:
            attrs_2d = img_attr[:, -1, :, :]

        else:
            raise ValueError(f"Unexpected img_attr shape: {img_attr.shape}")

        # 2. Upsample Heatmap
        attrs_upsampled = F.interpolate(
            attrs_2d.unsqueeze(1).float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # 3. Prepare Background
        rgb_background = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0

        print("\n" + "=" * 50)
        print("IMAGE ATTRIBUTIONS")
        print("=" * 50)

        cmap = self._get_cmap() if hasattr(self, "_get_cmap") else "coolwarm"

        for i in range(num_tokens):
            # --- NEW: Decode the exact matched token ---
            target_token_str = self.processor.decode([specific_target_ids[i]])
            print(f"\n[*] Heatmap for generated token: '{target_token_str}'")

            attr_map = attrs_upsampled[i].unsqueeze(-1).cpu().detach().numpy()

            _ = visualization.visualize_image_attr_multiple(
                attr_map,
                rgb_background,
                methods=["original_image", "blended_heat_map"],
                signs=["all", "positive"],
                show_colorbar=True,
                titles=[
                    "Original Image",
                    f"Attribution: '{target_token_str}'",
                    # "Overlay",
                ],
                use_pyplot=True,
                cmap=cmap,
            )

    def _plot_image_attributions(
        self,
        img_attr: Tensor,
        original_image: Image.Image,
        target_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        target_indices: int | list[int] | None = None,
        explainer_name: str = "Explainer",
        save_path: str | None = None,
    ):
        num_tokens = img_attr.shape[0]
        orig_w, orig_h = original_image.size
        # specific_target_ids = self._resolve_target_ids(target_ids, target_indices)

        # 1. Reshape Attributions
        if img_attr.dim() == 2:
            num_patches = img_attr.shape[1]
            if image_grid_thw is not None:
                grid_h = int(image_grid_thw[0, 1].item())
                grid_w = int(image_grid_thw[0, 2].item())
            else:
                grid_h = grid_w = int(np.sqrt(num_patches))
            attrs_2d = img_attr.view(num_tokens, grid_h, grid_w)
        elif img_attr.dim() == 3:
            attrs_2d = img_attr
        elif img_attr.dim() == 4:
            attrs_2d = img_attr[:, -1, :, :]
        else:
            raise ValueError(f"Unexpected img_attr shape: {img_attr.shape}")

        # 2. Upsample Heatmap
        attrs_upsampled = F.interpolate(
            attrs_2d.unsqueeze(1).float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # 3. Prepare Background
        rgb_background = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
        cmap = self._get_cmap() if hasattr(self, "_get_cmap") else "coolwarm"

        # We assume you only want to plot the heatmap for the FIRST generated token for the grid
        # If you want a specific token, ensure target_indices is passed correctly.
        # target_token_str = self.processor.decode([specific_target_ids[0]])
        attr_map = attrs_upsampled[0].unsqueeze(-1).cpu().detach().numpy()

        # 4. Generate the plot using Captum
        # Notice we only use "blended_heat_map" to save space in the LaTeX grid
        fig, ax = visualization.visualize_image_attr_multiple(
            attr_map,
            rgb_background,
            methods=["blended_heat_map"],
            signs=["positive"],  # usually 'positive' is best for text-to-image grounding
            show_colorbar=False,  # Disable colorbar so it fits nicely in the grid
            titles=[f"{explainer_name}"],
            use_pyplot=False,  # <--- Set to False so it doesn't block the loop
            cmap=cmap,
        )

        # 5. Save to disk if a path is provided
        if save_path:
            import matplotlib.pyplot as plt

            # bbox_inches='tight' removes annoying white margins
            fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
            plt.close(fig)  # Free up memory
            print(f"[+] Saved {explainer_name} heatmap to {save_path}")

    def save_combined_pdf(
        self,
        img_attr: Tensor,
        text_attr: Tensor,
        original_image,
        input_ids: Tensor,
        target_ids: Tensor,
        explainer_name: str,
        save_path: str,
        target_indices: int | list[int] | None = None,
        special_token_ids: list | None = None,
        semantic_mask: Tensor | None = None,
    ):
        """
        Creates a single PDF with the image heatmap on top and the text bar chart directly below it.
        """
        # ==========================================
        # 1. Setup the Matplotlib Figure (2 Rows, 1 Column)
        # ==========================================
        # height_ratios=[3, 1] makes the image 3 times taller than the bar chart
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), gridspec_kw={"height_ratios": [3, 1]})

        # ==========================================
        # 2. Process & Plot Image (Top Axis - axes[0])
        # ==========================================
        orig_w, orig_h = original_image.size
        # specific_target_ids = self._resolve_target_ids(target_ids, target_indices)

        # Reshape image attributions
        if img_attr.dim() == 2:
            num_patches = img_attr.shape[1]
            grid_h = grid_w = int(np.sqrt(num_patches))
            attrs_2d = img_attr.view(img_attr.shape[0], grid_h, grid_w)
        elif img_attr.dim() == 4:
            attrs_2d = img_attr[:, -1, :, :]
        else:
            attrs_2d = img_attr

        # Upsample to original image size
        attrs_upsampled = F.interpolate(
            attrs_2d.unsqueeze(1).float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(1)

        rgb_background = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
        cmap = self._get_cmap() if hasattr(self, "_get_cmap") else "coolwarm"
        attr_map = attrs_upsampled[0].unsqueeze(-1).cpu().detach().numpy()

        # Tell Captum to draw strictly on axes[0]
        visualization.visualize_image_attr(
            attr_map,
            rgb_background,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=False,
            title=explainer_name,
            plt_fig_axis=(fig, axes[0]),  # <--- THE SECRET SAUCE
            use_pyplot=False,
            cmap=cmap,
        )

        # ==========================================
        # 3. Process & Plot Text (Bottom Axis - axes[1])
        # ==========================================
        special_token_ids = special_token_ids or []
        input_ids_list = input_ids[0].tolist()

        valid_indices = [
            i
            for i, tok_id in enumerate(input_ids_list)
            if tok_id not in special_token_ids and (semantic_mask is None or semantic_mask[i].item())
        ]

        if valid_indices:
            filtered_input_ids = input_ids[0][valid_indices]
            txt_scores = text_attr[0, valid_indices].cpu().detach().numpy()

            # Normalize text scores
            max_abs = np.max(np.abs(txt_scores))
            if max_abs > 0:
                txt_scores = txt_scores / max_abs

            tokens = [self.processor.decode([tok]) for tok in filtered_input_ids]

            # Draw the bar chart on axes[1]
            colors = ["#1f77b4" if score > 0 else "#d62728" for score in txt_scores]
            axes[1].bar(range(len(tokens)), txt_scores, color=colors)

            # Clean formatting
            axes[1].set_xticks(range(len(tokens)))
            axes[1].set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
            axes[1].set_yticks([])  # Hide Y axis
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)
            axes[1].spines["left"].set_visible(False)
        else:
            axes[1].axis("off")  # Hide axis if no text

        # ==========================================
        # 4. Save the Unified PDF
        # ==========================================
        fig.tight_layout()  # Fixes overlapping text/titles
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def save_combined_highlight_pdf(
        self,
        img_attr: Tensor,
        text_attr: Tensor,
        original_image,
        input_ids: Tensor,
        target_ids: Tensor,
        explainer_name: str,
        save_path: str,
        target_indices: int | list[int] | None = None,
        special_token_ids: list | None = None,
        semantic_mask: Tensor | None = None,
    ):
        """
        Creates a single PDF with the image heatmap on top and inline colored text highlights below.
        """
        # ==========================================
        # 1. Setup the Matplotlib Figure (2 Rows, 1 Column)
        # ==========================================
        # height_ratios=[3, 1.2] gives the text area just enough room for 2-3 lines of wrapped text
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), gridspec_kw={"height_ratios": [3, 1.2]})

        # ==========================================
        # 2. Process & Plot Image (Top Axis - axes[0])
        # ==========================================
        orig_w, orig_h = original_image.size
        # specific_target_ids = self._resolve_target_ids(target_ids, target_indices)

        if img_attr.dim() == 2:
            num_patches = img_attr.shape[1]
            grid_h = grid_w = int(np.sqrt(num_patches))
            attrs_2d = img_attr.view(img_attr.shape[0], grid_h, grid_w)
        elif img_attr.dim() == 4:
            attrs_2d = img_attr[:, -1, :, :]
        else:
            attrs_2d = img_attr

        attrs_upsampled = F.interpolate(
            attrs_2d.unsqueeze(1).float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        ).squeeze(1)

        rgb_background = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0
        cmap_img = self._get_cmap() if hasattr(self, "_get_cmap") else "coolwarm"
        attr_map = attrs_upsampled[0].unsqueeze(-1).cpu().detach().numpy()

        # Draw image on axes[0]
        visualization.visualize_image_attr(
            attr_map,
            rgb_background,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=False,
            title=explainer_name,
            plt_fig_axis=(fig, axes[0]),
            use_pyplot=False,
            cmap=cmap_img,
        )

        # ==========================================
        # 3. Process & Plot Text Highlights (Bottom Axis - axes[1])
        # ==========================================
        axes[1].axis("off")  # Hide borders and ticks
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        special_token_ids = special_token_ids or []
        input_ids_list = input_ids[0].tolist()

        valid_indices = [
            i
            for i, tok_id in enumerate(input_ids_list)
            if tok_id not in special_token_ids and (semantic_mask is None or semantic_mask[i].item())
        ]

        if valid_indices:
            filtered_input_ids = input_ids[0][valid_indices]
            txt_scores = text_attr[0, valid_indices].cpu().detach().numpy()

            max_abs = np.max(np.abs(txt_scores))
            if max_abs > 0:
                txt_scores = txt_scores / max_abs

            tokens = [self.processor.decode([tok]) for tok in filtered_input_ids]

            # Setup Red-White-Green colormap
            cmap_text = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["#ff4d4d", "#ffffff", "#4dff4d"])

            # Draw the highlighted words with automatic wrapping
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            x_pos, y_pos = 0.0, 0.8
            line_spacing = 0.35

            for word, score in zip(tokens, txt_scores, strict=False):
                color_idx = (score + 1) / 2.0
                bg_color = cmap_text(color_idx)

                # Draw the text box
                t = axes[1].text(
                    x_pos,
                    y_pos,
                    word,
                    fontsize=9,
                    bbox={"facecolor": bg_color, "edgecolor": "none", "boxstyle": "round,pad=0.2", "alpha": 0.8},
                )

                # Calculate width to place the next word
                bbox = t.get_window_extent(renderer).transformed(axes[1].transData.inverted())
                x_pos += bbox.width + 0.02  # Add space between words

                # Drop to next line if we hit the right edge
                if x_pos > 0.95:
                    x_pos = 0.0
                    y_pos -= line_spacing

        # ==========================================
        # 4. Save the Unified PDF
        # ==========================================
        fig.tight_layout()
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
