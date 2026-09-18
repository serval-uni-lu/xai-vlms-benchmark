import math
from types import SimpleNamespace

import torch
import torch.nn as nn


class MockGridConfig:
    # Telling your pipeline this is LLaVA forces it to use standard 4D image logic
    model_type = "llava" 

class ToyGridVLM(nn.Module):
    """
    A pure mathematical Toy VLM. 
    Image is evaluated as a raw (H, W) spatial grid. 
    Text is evaluated as a 1D token sequence.
    """
    def __init__(self, model_wrapper, label):
        super().__init__()
        self.config = MockGridConfig()
        
        self.gt_pixel_mask = None   
        self.gt_text_mask = None    
        self.orig_pixel_values = None 
        self.target_ids = []

        self.model_wrapper = model_wrapper
        # Safely handle if label is multiple tokens
        self.target_vocab_id = self.model_wrapper.processor.tokenizer.encode(label)
        
        self.device = self.model_wrapper.device        
        self.pad_token_id = self.model_wrapper.processor.tokenizer.pad_token_id

        # A dummy layer to register parameters so .to(device) works if called
        self.lm_head = nn.Linear(1, 1)

    def set_sample_ground_truth(self, pixel_mask, text_mask, orig_pixel_values, target_ids):
        """Inject the masks and IMMEDIATELY move them to the correct device."""
        # FIX: Force everything to self.device right away!
        self.gt_pixel_mask = pixel_mask.bool().to(self.device)
        self.gt_text_mask = text_mask.bool().to(self.device)
        self.orig_pixel_values = orig_pixel_values.clone().to(self.device)
        
        if isinstance(target_ids, torch.Tensor):
            self.target_ids = target_ids.flatten().tolist()
        else:
            self.target_ids = list(target_ids)

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        device = self.device
        
        # ---------------------------------------------------------
        # 1. TEXT SYNERGY (1D)
        # ---------------------------------------------------------
        if input_ids is not None:
            prompt_len = self.gt_text_mask.shape[0]
            slice_len = min(input_ids.shape[1], prompt_len)
            
            # input_ids comes in on GPU. sliced_input_ids stays on GPU.
            sliced_input_ids = input_ids[0, :slice_len]
            active_prompt_tokens = (sliced_input_ids != self.pad_token_id)
            
            if slice_len < prompt_len:
                padding = torch.zeros(prompt_len - slice_len, dtype=torch.bool, device=device)
                active_prompt_tokens = torch.cat([active_prompt_tokens, padding])
        else:
            # FIX: Because gt_text_mask is on GPU, this is now safely on GPU
            active_prompt_tokens = torch.ones_like(self.gt_text_mask)

        # ---------------------------------------------------------
        # 2. IMAGE SYNERGY (2D Spatial Grid)
        # ---------------------------------------------------------
        if pixel_values is not None and self.orig_pixel_values is not None:
            # pixel_values shape: (1, 3, H, W)
            pixel_values = pixel_values.to(device)
            # Both tensors are now safely on the same device
            diff = (pixel_values[0] - self.orig_pixel_values[0]).abs().mean(dim=0)
            active_pixels = (diff < 1e-4) # Shape: (H, W)
        else:
            # FIX: Safely created on GPU
            active_pixels = torch.ones_like(self.gt_pixel_mask)

        # ---------------------------------------------------------
        # 3. CALCULATE JOINT SYNERGY SCORE
        # ---------------------------------------------------------
        total_true_tokens = self.gt_text_mask.sum().float()
        true_tokens_present = (active_prompt_tokens & self.gt_text_mask).sum().float()
        text_score = (true_tokens_present / total_true_tokens).item() if total_true_tokens > 0 else 1.0
        
        total_true_pixels = self.gt_pixel_mask.sum().float()
        true_patches_present = (active_pixels & self.gt_pixel_mask).sum().float()
        visual_score = (true_patches_present / total_true_pixels).item() if total_true_pixels > 0 else 1.0
        
        synergy = text_score * visual_score

        # ---------------------------------------------------------
        # 4. BULLETPROOF LOGIT INJECTION
        # ---------------------------------------------------------
        N = max(1, len(self.target_ids))
        P_joint = max(synergy, 1e-7)
        P_joint = min(P_joint, 1.0 - 1e-7)
        
        P_token = math.pow(P_joint, 1.0 / N)
        vocab_size = 151936
        target_logit = math.log(P_token / (1.0 - P_token)) + math.log(vocab_size - 1)
        
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10
        logits = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        
        for tid in self.target_ids:
            logits[:, :, tid] = target_logit

                    
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        device = self.device
        fake_new_ids = torch.tensor(self.target_ids if self.target_ids else [42], device=device)
        fake_full_ids = torch.cat([kwargs.get("input_ids", torch.tensor([[0]], device=device))[0], fake_new_ids])
        return SimpleNamespace(sequences=fake_full_ids.unsqueeze(0), hidden_states=(torch.zeros(1,1,1, device=device),))

