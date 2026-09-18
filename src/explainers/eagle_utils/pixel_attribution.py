import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm


class UniversalEAGLEVisionAdaptor(nn.Module):
    """
    Bridges EAGLE's Vision Submodular Explainer with your BaseVLMWrapper,
    incorporating all necessary sequence alignment and stale-key removal steps.
    """
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper # Your BaseVLMWrapper instance
        self.device = wrapper.device
        self.softmax = nn.Softmax(dim=-1)
        
        # 1. Empty 2D Tensor (Batch, Seq_Len) for token IDs
        # Using dtype=torch.long because token IDs are always integers
        self.generated_ids: torch.Tensor = torch.empty((0, 0), dtype=torch.long)
        
        # 2. Empty 1D NumPy array for absolute sequence positions
        self.target_token_position: np.ndarray = np.array([], dtype=np.int64)
        
        # 3. Empty Python list for the vocabulary word integers
        self.selected_interpretation_token_word_id: list[int] = []
        
        # 4. Empty string for the raw text prompt
        self.text_prompt: str = ""

    def forward(self, image):
        """
        Processes the masked image from EAGLE, rebuilds the prompt/answer context, 
        and extracts the specific token probabilities.
        """
        # 1. EAGLE occasionally passes raw tensors from cv2; format to PIL if needed
        if isinstance(image, torch.Tensor) and image.shape[-1] == 3:
            image_tensor = image[..., [2, 1, 0]]  # BGR → RGB
            image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.clamp(0, 255).byte()
            image = TF.to_pil_image(image_tensor)

        # 2. Get base inputs using your wrapper's unified method
        # This replaces the hardcoded processor.apply_chat_template
        inputs = self.wrapper.get_inputs(image=image, text=self.text_prompt)

        # 3. Truncate to max target position (Matches your Qwen code)
        max_pos = int(max(self.target_token_position))
        forced_input_ids = self.generated_ids[:, :max_pos]
        
        inputs['input_ids'] = forced_input_ids
        inputs['attention_mask'] = torch.ones_like(forced_input_ids)

        # 4. Handle Qwen-specific multimodal token padding
        # We wrap this in a safe check so it doesn't break non-Qwen VLMs
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"]
            if mm_token_type_ids.shape[1] < forced_input_ids.shape[1]:
                text_tail = torch.zeros(
                    (mm_token_type_ids.shape[0], forced_input_ids.shape[1] - mm_token_type_ids.shape[1]),
                    dtype=mm_token_type_ids.dtype,
                    device=mm_token_type_ids.device
                )
                mm_token_type_ids = torch.cat([mm_token_type_ids, text_tail], dim=1)
            inputs["mm_token_type_ids"] = mm_token_type_ids[:, :forced_input_ids.shape[1]]

        # 5. Remove stale keys so the model recomputes positional embeddings
        for stale_key in ("token_type_ids", "position_ids", "cache_position"):
            inputs.pop(stale_key, None)

        # Ensure everything is on the correct device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 6. Unified Forward Pass through your wrapper
        with torch.no_grad():
            # Requires your wrapper to return the full sequence of logits
            all_logits = self.wrapper(
                **inputs,
                return_full_sequence=True 
            )

        # 7. Extract Logits and Probabilities
        # if self.generated_ids is not None:
        # Shift by -1 because the generated content relies on the previous position
        returned_logits = all_logits[:, self.target_token_position - 1]
        returned_logits = self.softmax(returned_logits)
        
        if self.selected_interpretation_token_word_id is not None:
            # Safely convert target words to tensor
            word_ids = torch.as_tensor(
                self.selected_interpretation_token_word_id, 
                device=self.device
            )
            indices = word_ids.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
            
            # Gather specific probabilities
            returned_logits = returned_logits.gather(dim=2, index=indices)
            returned_logits = returned_logits.squeeze(-1) # [1, N]
                
        return returned_logits[0]


class MLLMSubModularExplanationVision:
    """
    Black-box explanation of multimodal large language 
    model (MLLM) based on submodular subset selection.
    """
    def __init__(self, 
                 model,
                 preproccessing_function = None,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                #  batch_size = 4,    # Suggestion: 
                 ):
        """_summary_

        Args:
            detection_model (_type_): Any Multimodal Large Language Models
            preproccessing_function (_type_): _description_
            lambda1 (float, optional): _description_. Defaults to 1.0.
            lambda2 (float, optional): _description_. Defaults to 1.0.
            device (str, optional): _description_. Defaults to "cuda".
        """
        super().__init__()
        # Parameters of the submodular
        self.MLLM = model
        self.preproccessing_function = preproccessing_function
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.device = self.MLLM.device
        
        print(self.device)
        
        # self.batch_size = batch_size
        
    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["insertion_score"] = []
        self.saved_json_file["deletion_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["insertion_word_score"] = []
        self.saved_json_file["deletion_word_score"] = []
        self.saved_json_file["region_area"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
    
    def MLLM_inference_batch_images(self, images):
        """_summary_

        Args:
            images (Tensor): _description_
        """
        results = []
        for image in images:
            output_logits = self.MLLM(image)
            results.append(output_logits)
            
        results = torch.stack(results, dim=0)
        return results
    
    def evaluation_maximun_sample(self, S_set):
        # timer = time.time()
        # V_set_tem = np.array(self.V_set).astype(np.float32) # (51, 1365, 2048, 1)
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)
        
        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)
        alpha_batch = alpha_batch.expand(-1, -1, -1, 3)
        
        # print("Stage 1 time comsume: {}".format(time.time()-timer))
        # timer = time.time()
        
        source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0], -1, -1, -1)
        batch_input_images = alpha_batch * source_tensor    # torch.Size([51, 1365, 2048, 3])
        batch_input_images_reverse = (1 - alpha_batch) * source_tensor
        
        # print("Stage 2 time comsume: {}".format(time.time()-timer))
        # timer = time.time()
        
        with torch.no_grad():
            # Insertion
            insertion_scores = self.MLLM_inference_batch_images(batch_input_images
                                                                ).to(torch.float32)
            
            # Deletion
            deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse
                                                               ).to(torch.float32)
            
            #Overall submodular score
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1-deletion_scores)
            smdl_scores = smdl_scores.mean(-1)
            arg_max_index = smdl_scores.argmax().cpu().item()
            
            # Save intermediate results
            self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].mean()
                                                           .cpu().numpy().item())
            self.saved_json_file["insertion_word_score"].append(insertion_scores[arg_max_index]
                                                                .cpu().numpy().tolist())
            
            self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].mean()
                                                          .cpu().numpy().item())
            self.saved_json_file["deletion_word_score"].append(deletion_scores[arg_max_index]
                                                               .cpu().numpy().tolist())
            
            self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())
            
            # Update
            S_set.append(self.V_set[arg_max_index])
            self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]
                                                                         ).float().to(self.device)
            del self.V_set[arg_max_index]
            
            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )
            
        return S_set
    
    def get_merge_set(self):
        # define a subset
        S_set = []
        # self.refer_baseline = np.zeros_like(self.V_set[0]).astype(np.float32)
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)
        
        for _i in tqdm(range(self.saved_json_file["sub-region_number"])):
            S_set = self.evaluation_maximun_sample(S_set)
        
        self.saved_json_file["org_score"] = self.saved_json_file["insertion_word_score"][-1]
        self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_word_score"][-1]
        
        return S_set
    
    def __call__(self, image, V_set):
        """_summary_

        Args:
            image (cv2 format): (h, w, 3)
            V_set (_type_): (n, h, w, 1)
            class_id (List [int, ...]): which classes?
            given_box (xyxy): which boxes?
        """
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)
        
        self.source_image = image
        self.source_tensor = torch.from_numpy(self.source_image).float().to(self.device)
        self.h, self.w, _ = self.source_image.shape
        self.region_area = image.shape[0] * image.shape[1]
        
        self.V_set = V_set.copy()
        
        Submodular_Subset = self.get_merge_set()
        
        return Submodular_Subset, self.saved_json_file


class EfficientMLLMSubModularExplanationVision(MLLMSubModularExplanationVision):
    """
    Black-box explanation of multimodal large language 
    model (MLLM) based on submodular subset selection.
    """
    def __init__(self, 
                 model: UniversalEAGLEVisionAdaptor,
                 preproccessing_function = None,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 search_scope = 10,
                 pending_samples = 8,
                 update_step = 5,
                #  batch_size = 4,    # Suggestion: 
                 ):
        """_summary_

        Args:
            detection_model (_type_): Any Multimodal Large Language Models
            preproccessing_function (_type_): _description_
            lambda1 (float, optional): _description_. Defaults to 1.0.
            lambda2 (float, optional): _description_. Defaults to 1.0.
            device (str, optional): _description_. Defaults to "cuda".
        """
        super().__init__(
            model = model,
            preproccessing_function = preproccessing_function,
            lambda1 = lambda1,
            lambda2 = lambda2,
        )
        # Parameters of the submodular
        self.device = self.MLLM.device

        self.search_scope = search_scope
        self.update_step = update_step
        
        self.pending_samples = pending_samples
    
    def save_file_init(self):
        self.saved_json_file = {
            "insertion_score": [],
            "deletion_score": [],
            "smdl_score": [],
            "insertion_word_score": [],
            "deletion_word_score": [],
            "region_area": [],
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "insertion_score_oppose": [],
            "deletion_score_oppose": [],
            "insertion_word_score_oppose": [],
            "deletion_word_score_oppose": [],
            "region_area_oppose": [],
        }
        
    def save_positive_file(self, insertion_score, deletion_score, smdl_score):
        self.saved_json_file["insertion_score"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score"].append(insertion_score.cpu().numpy().tolist())
        
        self.saved_json_file["deletion_score"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score"].append(deletion_score.cpu().numpy().tolist())
        
        self.saved_json_file["smdl_score"].append(smdl_score.cpu().item())
        
    def save_negative_file(self, insertion_score, deletion_score):
        self.saved_json_file["insertion_score_oppose"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score_oppose"].append(insertion_score.cpu().numpy().tolist())
        
        self.saved_json_file["deletion_score_oppose"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score_oppose"].append(deletion_score.cpu().numpy().tolist())
    
    def evaluation_maximun_sample(self, S_set, S_set_opposite):
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)
        
        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)
        alpha_batch = alpha_batch.expand(-1, -1, -1, 3)

        if len(S_set) == 0 or self.update_count % self.update_step == 0:
            # Positive samples search
            source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0],
                                                                   -1, -1, -1)

        else:
            alpha_batch = alpha_batch[:self.search_scope]
            # Positive samples search with scope
            source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch.shape[0],
                                                                   -1, -1, -1)

        batch_input_images = alpha_batch * source_tensor    # torch.Size([51, 1365, 2048, 3])
        batch_input_images_reverse = (1 - alpha_batch) * source_tensor
        
        with torch.no_grad():
            # Insertion
            insertion_scores = self.MLLM_inference_batch_images(batch_input_images
                                                                ).to(torch.float32)
            
            # Deletion
            deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse
                                                               ).to(torch.float32)
            
            #Overall submodular score
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1-deletion_scores)
            smdl_scores = smdl_scores.mean(-1)
            arg_max_index = smdl_scores.argmax().cpu().item()
            
            if len(S_set) == 0 or self.update_count % self.update_step == 0:
                indices = torch.argsort(smdl_scores, descending=True)
                sorted_V = [self.V_set[i] for i in indices]
                self.V_set = sorted_V
                
                # Update 0 -> Have been sorted
                S_set.append(self.V_set[0])
                self.refer_baseline = self.refer_baseline \
                    + torch.from_numpy(self.V_set[0]).float().to(self.device)
                del self.V_set[0]
                
            else:
                # Update 0 -> Have been sorted
                S_set.append(self.V_set[arg_max_index])
                self.refer_baseline = self.refer_baseline \
                    + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
                del self.V_set[arg_max_index]
                
            # Save intermediate results
            self.save_positive_file(
                insertion_scores[arg_max_index], 
                deletion_scores[arg_max_index], 
                smdl_scores[arg_max_index])
            
            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )
            
            # Negative samples search
            if len(S_set_opposite) == 0:
                arg_min_index = smdl_scores.argmin().cpu().item()
                
                self.save_negative_file(
                    deletion_scores[arg_min_index], 
                    insertion_scores[arg_min_index])
                
                S_set_opposite.append(self.V_set[-1])
                self.refer_baseline_opposite = self.refer_baseline_opposite \
                    - torch.from_numpy(self.V_set[-1]).float().to(self.device)
                del self.V_set[-1]
                
                self.saved_json_file["region_area_oppose"].append(
                    (self.refer_baseline_opposite.sum() / self.region_area).cpu().item()
                )
                
                return S_set, S_set_opposite
            
            elif len(self.V_set) > self.pending_samples:
                oppose_set_condidates = self.V_set[-self.pending_samples:]
                
                oppose_set_tensor = torch.from_numpy(np.array(oppose_set_condidates)
                                                     ).float().to(self.device)
                
                alpha_batch_opposite = self.refer_baseline_opposite.unsqueeze(0) \
                    - oppose_set_tensor
                alpha_batch_opposite = alpha_batch_opposite.expand(-1, -1, -1, 3)
                
                source_tensor = self.source_tensor.unsqueeze(0).expand(alpha_batch_opposite.shape[0], -1, -1, -1)
                batch_input_images = alpha_batch_opposite * source_tensor    # torch.Size([51, 1365, 2048, 3])
                batch_input_images_reverse = (1 - alpha_batch_opposite) * source_tensor
                
                # Insertion
                insertion_scores = self.MLLM_inference_batch_images(batch_input_images).to(torch.float32)
                # Deletion
                deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse).to(torch.float32)
                
                oppose_smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1-deletion_scores)
                
                arg_max_index_oppose = oppose_smdl_scores.mean(-1).argmax().cpu().item()
                
                self.save_negative_file(
                    insertion_scores[arg_max_index_oppose], 
                    deletion_scores[arg_max_index_oppose])
                
                S_set_opposite.append(oppose_set_condidates[arg_max_index_oppose])
                self.refer_baseline_opposite = self.refer_baseline_opposite - \
                torch.from_numpy(oppose_set_condidates[arg_max_index_oppose]).float().to(self.device)
            
                del self.V_set[-self.pending_samples + arg_max_index_oppose]
                
                self.saved_json_file["region_area_oppose"].append(
                    (self.refer_baseline_opposite.sum() / self.region_area).cpu().item()
                )
            
        return S_set, S_set_opposite
    
    def get_merge_set(self):
        # define a subset
        S_set = []
        S_set_opposite = []
        
        # self.refer_baseline = np.zeros_like(self.V_set[0]).astype(np.float32)
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(),
                                               device=self.device)
        self.refer_baseline_opposite = 1 - torch.zeros_like(torch.from_numpy(self.V_set[0]).float(),
                                                            device=self.device)
        
        sub_images = torch.stack([self.source_tensor * self.refer_baseline,
                                  self.source_tensor * self.refer_baseline_opposite])
        scores = self.MLLM_inference_batch_images(sub_images).to(torch.float32)
        
        self.saved_json_file["org_score"] = scores[1].cpu().numpy().tolist()
        self.saved_json_file["baseline_score"] = scores[0].cpu().numpy().tolist()
        
        self.update_count = 0
        for _i in tqdm(range(self.saved_json_file["sub-region_number"])):
            if len(self.V_set) == 1:
                S_set = S_set + self.V_set
                break
            S_set, S_set_opposite = self.evaluation_maximun_sample(S_set, S_set_opposite)
            self.update_count += 1
        
        self.saved_json_file["insertion_score"] = self.saved_json_file["insertion_score"] + \
        self.saved_json_file["insertion_score_oppose"][::-1] + [scores[1].cpu().mean().item()]
        self.saved_json_file["deletion_score"] = self.saved_json_file["deletion_score"] + \
        self.saved_json_file["deletion_score_oppose"][::-1] + [scores[0].cpu().mean().item()]
        self.saved_json_file["insertion_word_score"] = self.saved_json_file["insertion_word_score"] + \
        self.saved_json_file["insertion_word_score_oppose"][::-1] + [scores[1].cpu().numpy().tolist()]
        self.saved_json_file["deletion_word_score"] = self.saved_json_file["deletion_word_score"] + \
        self.saved_json_file["deletion_word_score_oppose"][::-1] + [scores[0].cpu().numpy().tolist()]
        
        self.saved_json_file["region_area"] = self.saved_json_file["region_area"] \
            + self.saved_json_file["region_area_oppose"][::-1] + [1.]
        S_set = np.concatenate((S_set, S_set_opposite[::-1]))
        
        return S_set
    
    def __call__(self, image, V_set):
        """_summary_

        Args:
            image (cv2 format): (h, w, 3)
            V_set (_type_): (n, h, w, 1)
            class_id (List [int, ...]): which classes?
            given_box (xyxy): which boxes?
        """
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)
        
        self.source_image = image
        self.source_tensor = torch.from_numpy(self.source_image).float().to(self.device)
        self.h, self.w, _ = self.source_image.shape
        self.region_area = image.shape[0] * image.shape[1]
        
        self.V_set = V_set.copy()
        
        Submodular_Subset = self.get_merge_set()
        self.saved_json_file["smdl_score"] = (np.array(self.saved_json_file["insertion_score"]) +
                                              1 - np.array(self.saved_json_file["deletion_score"])).tolist()
        
        return Submodular_Subset, self.saved_json_file

