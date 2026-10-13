import string
import unicodedata

import torch
from sentence_transformers import SentenceTransformer, util


class OntologyMapper:
    def __init__(self, coco_categories, threshold=0.5, device=None):
        """
        Args:
            coco_categories (dict): {id: "name"} e.g., {3: "car", 4: "motorcycle"}
            threshold (float): Minimum cosine similarity to accept a match.
        """
        # Load a tiny, blazingly fast embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.threshold = threshold

        self.category_ids = list(coco_categories.keys())
        self.category_names = list(coco_categories.values())

        # Pre-compute embeddings for the 80 COCO classes (only happens once)
        self.coco_embeddings = self.model.encode(self.category_names, convert_to_tensor=True)

    def map_word(self, word):
        """Maps an open-vocabulary word to the closest COCO category ID."""
        # Embed the generated word
        word_embedding = self.model.encode(word, convert_to_tensor=True)

        # Compute Cosine Similarity against all COCO classes
        cos_scores = util.cos_sim(word_embedding, self.coco_embeddings)[0]
        # print(cos_scores)
        # print(cos_scores.shape)
        # Find the best match
        best_score, best_idx = torch.max(cos_scores, dim=0)
        # print(best_idx)

        if best_score.item() >= self.threshold:
            return self.category_ids[best_idx]

        return None  # Word is too abstract or not in COCO (e.g., "running")


def is_english_punctuation(char):
    return char in string.punctuation


def is_chinese_char_or_punctuation(char):
    # for ch in char:
    #     if "CJK" in unicodedata.name(ch, ""):
    #         return True
    return any("CJK" in unicodedata.name(ch, "") for ch in char)


def ids_to_word_groups(ids, processor):

    txt = processor.batch_decode(ids)[0]
    tokens = processor.tokenizer.tokenize(txt)
    words, tokens_idx = [], []
    for i, _ in enumerate(tokens):
        word = processor.tokenizer.decode(processor.tokenizer.convert_tokens_to_ids(_))
        if (
            i == 0
            or is_english_punctuation(word)
            or is_chinese_char_or_punctuation(word)
            or word[0] == " "
            or _[0] == "▁"
        ):
            words.append(word.replace(" ", ""))
            tokens_idx.append([i])
        else:
            words[-1] += word.replace(" ", "")
            tokens_idx[-1].append(i)
    return words, tokens_idx


def pool_heatmaps(heatmaps_tensor, token_indices, method="max"):
    """
    Pools the heatmaps for a specific word that was split into multiple tokens.

    Args:
        heatmaps_tensor (torch.Tensor): The full sequence attribution of shape (Seq_Len, H, W).
        token_indices (list): The list of token indices for the target word.
        method (str): 'max' or 'mean'.

    Returns:
        torch.Tensor: Pooled heatmap of shape (H, W).
    """
    # Extract just the heatmaps for the subwords of this specific word
    subword_heatmaps = heatmaps_tensor[token_indices]  # Shape: (Num_Subwords, H, W)

    # If the word was just a single token, return it as-is
    if len(token_indices) == 1:
        return subword_heatmaps.squeeze(0)
    print(subword_heatmaps.shape)

    # If it was split, pool it
    if method == "max":
        # Max pooling across the subword dimension (Standard in XAI)
        pooled_heatmap, _ = torch.max(subword_heatmaps, dim=0)
    elif method == "mean":
        # Mean pooling across the subword dimension
        pooled_heatmap = torch.mean(subword_heatmaps, dim=0)
    else:
        raise ValueError("Method must be 'max' or 'mean'")

    return pooled_heatmap
