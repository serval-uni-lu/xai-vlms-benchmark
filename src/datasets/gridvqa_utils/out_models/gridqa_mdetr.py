"""
GridQA MDETR Model Wrapper

This module provides a wrapper class for the MDETR model on the GridQA dataset,
enabling use with the MultiViz analysis framework.

The model handles visual question answering on grid-based images, supporting
both counting questions (global head) and yes/no questions (obj head).
"""

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from torchvision import transforms

from torch.nn.attention import sdpa_kernel, SDPBackend

# Make this package's modules importable when scripts are run from the
# explainability/ root, regardless of the caller's working directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)  # so `from analysismodel import ...` resolves

# The patched MDETR checkout (see ../mdetr/README.md) must be importable as the
# `mdetr` package. Either install it, add it to PYTHONPATH, or set MDETR_ROOT.
_MDETR_ROOT = os.environ.get("MDETR_ROOT")
if _MDETR_ROOT:
    sys.path.append(_MDETR_ROOT)

from .analysismodel import analysismodel
from mdetr.models import mdetr as mdetr_module

def build_mdetr_args(device_str: str = "cuda", input_size: int = 320):
    """
    Build arguments namespace for MDETR model construction.
    
    Parameters
    ----------
    device_str : str
        Device to use ('cuda' or 'cpu').
    input_size : int
        Input image size (default 320 for GridQA).
    
    Returns
    -------
    SimpleNamespace
        Arguments for MDETR model building.
    """
    args = SimpleNamespace()
    args.device = device_str
    args.dataset_name = "gridvqa"

    # Backbone configuration (resnet101 for stability)
    args.backbone = "resnet101"
    args.backbone_name = args.backbone
    args.dilation = False
    args.pretrained_backbone = True
    args.lr_backbone = 1e-6
    args.masks = False
    args.position_embedding = "sine"
    args.input_image_size = input_size

    # Transformer architecture
    args.hidden_dim = 256
    args.enc_layers = 6
    args.dec_layers = 6
    args.nheads = 8
    args.dim_feedforward = 2048
    args.dropout = 0.1
    args.activation = "relu"
    args.pre_norm = False
    args.normalize_before = False
    args.return_intermediate_dec = True
    args.pass_pos_and_query = True
    args.num_queries = 100

    # QA heads configuration
    args.do_qa = True
    args.qa_dataset = "gqa"  # Use GQA structure for QA heads
    args.split_qa_heads = True
    args.predict_final = False
    args.qa_loss_coef = 1.0
    args.no_detection = False

    # Loss / matcher configuration
    args.set_loss = "hungarian"
    args.set_cost_class = 1.0
    args.set_cost_bbox = 5.0
    args.set_cost_giou = 2.0
    args.eos_coef = 0.1
    args.ce_loss_coef = 1.0
    args.bbox_loss_coef = 5.0
    args.giou_loss_coef = 2.0
    args.aux_loss = True

    # Contrastive loss configuration
    args.contrastive_loss = True
    args.contrastive_align_loss = True
    args.contrastive_loss_hdim = 64
    args.contrastive_loss_coef = 0.1
    args.contrastive_align_loss_coef = 1.0
    args.temperature_NCE = 0.07

    # Text encoder configuration
    args.text_encoder_type = "roberta-base"
    args.freeze_text_encoder = False
    args.text_encoder_pretrained = True

    # Mask / segmentation defaults
    args.mask_model = "none"
    args.mask_loss_coef = 1.0
    args.dice_loss_coef = 1.0

    # Additional required arguments for MDETR build
    # Need to include "gqa" for QA heads to be initialized properly
    args.combine_datasets = ["gqa"]
    args.combine_datasets_val = ["gqa"]

    return args


class GridQAMDETR(analysismodel):
    """
    MDETR model wrapper for GridQA dataset analysis.
    
    This class wraps the MDETR model for use with the MultiViz analysis
    framework, enabling gradient-based analysis, LIME, and other
    explainability methods on GridQA visual question answering.
    
    Parameters
    ----------
    device : str
        Device to use ('cuda' or 'cpu').
    checkpoint_path : str, optional
        Path to a pretrained checkpoint. If None, uses randomly initialized weights.
    input_size : int
        Input image size (default 320).
    """

    def __init__(self, device="cuda", checkpoint_path=None, input_size=320):
        super(analysismodel, self).__init__()
        self.device = device
        self.input_size = input_size
        
        # Modality configuration
        self.modalitynames = ["image", "text"]
        self.modalitytypes = ["image", "text"]
        
        # Build MDETR model
        build_args = build_mdetr_args(device, input_size)
        
        # Build model using the imported mdetr_module.build function
        # Returns: (model, criterion, contrastive_criterion, qa_criterion, weight_dict)
        self.model, _, _, _, _ = mdetr_module.build(build_args)
        
        self.model.to(device)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Image normalization transform
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Answer mappings for GridQA
        # Global head outputs counts 0-16 (17 classes for counting)
        # Obj head outputs yes/no (2 classes)
        self.count_answers = [str(i) for i in range(17)] + ["17+"]
        self.yesno_answers = ["yes", "no"]
        
        # Combined answer mapping (for compatibility)
        # Indices 0-17: counts, 18-19: yes/no
        self.answermapping = self.count_answers + self.yesno_answers

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a checkpoint file."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', 
                         checkpoint.get('model', checkpoint))
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            cleaned_state_dict[name] = v
        
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        print("Checkpoint loaded successfully.")

    def _preprocess_image(self, image_input):
        """
        Preprocess image input to tensor.
        
        Parameters
        ----------
        image_input : str, np.ndarray, or PIL.Image
            Image as file path, numpy array, or PIL Image.
        
        Returns
        -------
        torch.Tensor
            Preprocessed image tensor of shape (1, 3, H, W).
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        return self.transform(image).unsqueeze(0).to(self.device)

    def _interpret_qa_output(self, outputs):
        """
        Interpret QA head outputs to get predicted answer.
        
        Parameters
        ----------
        outputs : dict
            Model output dictionary containing QA predictions.
        
        Returns
        -------
        tuple
            (answer_str, answer_idx, answer_type)
            - answer_str: Human-readable answer
            - answer_idx: Index in answermapping
            - answer_type: 'count' or 'yesno'
        """
        # Determine answer type from pred_answer_type
        # Type 0: obj (yes/no), Type 3: global (count)

        pred_answer_type = outputs["pred_answer_type"].argmax(-1).item()  # 0: obj, 3: global
        if pred_answer_type == 0:  # obj head (yes/no)
            pred_obj = outputs["pred_answer_obj"].argmax(-1).item()
            return "yes" if pred_obj == 0 else "no"
        elif pred_answer_type == 3:  # global head (count)
            pred_global = outputs["pred_answer_global"].argmax(-1).item()
            return f"{pred_global}" if pred_global < 17 else "17+"
        else:  # Fallback
            yes_no_conf = torch.softmax(outputs["pred_answer_obj"], dim=-1).max().item()
            count_conf = torch.softmax(outputs["pred_answer_global"], dim=-1).max().item()
            if yes_no_conf > count_conf:
                pred_obj = outputs["pred_answer_obj"].argmax(-1).item()
                return f"yes? (conf:{yes_no_conf:.2f})" if pred_obj == 0 else f"no? (conf:{yes_no_conf:.2f})"
            else:
                pred_global = outputs["pred_answer_global"].argmax(-1).item()
                count_str = f"{pred_global}" if pred_global < 17 else "17+"
                return f"{count_str}? (conf:{count_conf:.2f})"

    def getunimodaldata(self, datainstance, modality):
        """
        Get unimodal data from a data instance.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance from GridQADataset.
            If tuple: (image, question, answer, label)
            If dict: full data dict from getdata_full()
        modality : str
            'image' or 'text'
        
        Returns
        -------
        np.ndarray or str
            Image array or question text.
        """
        if isinstance(datainstance, dict):
            if modality == "image":
                return datainstance['image']
            elif modality == "text":
                return datainstance['question']
        else:
            # Tuple format: (image, question, answer, label)
            if modality == "image":
                img = datainstance[0]
                if isinstance(img, str):
                    return np.asarray(Image.open(img).convert("RGB"))
                return img
            elif modality == "text":
                return datainstance[1]
        
        raise ValueError(f"Unknown modality: {modality}")

    def getcorrectlabel(self, datainstance):
        """
        Get the correct label for a data instance.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance from GridQADataset.
        
        Returns
        -------
        int
            Correct answer (typically a count for GridQA).
        """
        if isinstance(datainstance, dict):
            return datainstance['answer']

    def forward(self, datainstance):
        """
        Run forward pass on a single data instance.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance containing image and question.
        
        Returns
        -------
        dict
            Result object containing logits and predictions.
        """
        # Extract image and question
        if isinstance(datainstance, dict):
            image = datainstance['image']
            question = datainstance['question']
        else:
            image = datainstance[0]
            question = datainstance[1]
        
        # Preprocess
        img_tensor = self._preprocess_image(image)
        captions = [question]
        
        memory_cache = self.model(img_tensor, captions, encode_and_save=True)
        outputs = self.model(img_tensor, captions, encode_and_save=False, 
                                memory_cache=memory_cache)
        
        # Concatenate QA head outputs for analysis
        # pred_answer_obj: (B, 2) - yes/no
        # pred_answer_attr: (B, N_attr)
        # pred_answer_rel: (B, N_rel)
        # pred_answer_global: (B, 17) - counts
        # pred_answer_cat: (B, N_cat)
        
        qa_logits = {
            'pred_answer_obj': outputs['pred_answer_obj'].cpu(),
            'pred_answer_global': outputs['pred_answer_global'].cpu(),
            'pred_answer_type': outputs['pred_answer_type'].cpu(),
        }
        
        # Also store detection outputs
        result = {
            'qa_logits': qa_logits,
            'pred_logits': outputs['pred_logits'].cpu(),
            'pred_boxes': outputs['pred_boxes'].cpu(),
            'outputs_raw': outputs,  # Keep for advanced analysis
        }

        # Interpret answer
        answer_str, answer_idx, answer_type = self._interpret_qa_output(outputs)
        result['pred_answer_str'] = answer_str
        result['pred_answer_idx'] = answer_idx
        result['pred_answer_type'] = answer_type
        
        return result

    def forwardbatch(self, datainstances, batch_size=8):
        """
        Run forward pass on a batch of data instances with true batching.
        
        Parameters
        ----------
        datainstances : list
            List of data instances.
        batch_size : int
            Number of instances to process in each batch (default 8).
        
        Returns
        -------
        list
            List of result objects.
        """
        results = []
        num_instances = len(datainstances)
        
        with torch.no_grad():
            for batch_start in range(0, num_instances, batch_size):
                batch_end = min(batch_start + batch_size, num_instances)
                batch = datainstances[batch_start:batch_end]
                
                # Collect images and captions for the batch
                img_tensors = []
                captions = []
                
                for di in batch:
                    image = di['image']
                    question = di['question']
                    
                    img_tensors.append(self._preprocess_image(image))
                    captions.append(question)
                
                # Stack images into a single batch tensor
                batched_images = torch.cat(img_tensors, dim=0)
                
                # Forward pass with batched inputs
                memory_cache = self.model(batched_images, captions, encode_and_save=True)
                outputs = self.model(batched_images, captions, encode_and_save=False,
                                    memory_cache=memory_cache)
                
                # Process outputs for each item in the batch
                batch_results = self._process_batch_outputs(outputs, len(batch))
                results.extend(batch_results)
        
        return results

    def forward_tensor_batch(self, image_tensors: torch.Tensor, captions: list):
        """Run forward pass on pre-processed image tensors (skip PIL/transform).
        
        Parameters
        ----------
        image_tensors : torch.Tensor
            Batch of already-normalised image tensors, shape (B, 3, H, W),
            already on the correct device.
        captions : list[str]
            One caption per image.
        
        Returns
        -------
        list[dict]
            Per-instance QA output dicts (same format as forwardbatch).
        """
        results = []
        B = image_tensors.shape[0]
        bs = min(B, 256)  # internal sub-batch to avoid OOM on huge batches
        with torch.no_grad():
            for start in range(0, B, bs):
                end = min(start + bs, B)
                batch_imgs = image_tensors[start:end]
                batch_caps = captions[start:end]
                memory_cache = self.model(batch_imgs, batch_caps, encode_and_save=True)
                outputs = self.model(batch_imgs, batch_caps, encode_and_save=False,
                                     memory_cache=memory_cache)
                results.extend(self._process_batch_outputs(outputs, end - start))
        return results

    def _process_batch_outputs(self, outputs, batch_size):
        """
        Process batched model outputs into individual result objects.
        
        Parameters
        ----------
        outputs : dict
            Batched model output dictionary.
        batch_size : int
            Number of instances in the batch.
        
        Returns
        -------
        list
            List of result dictionaries, one per instance.
        """
        results = []
        
        for i in range(batch_size):
            # Extract per-instance outputs
            instance_outputs = {
                'pred_answer_obj': outputs['pred_answer_obj'][i:i+1],
                'pred_answer_global': outputs['pred_answer_global'][i:i+1],
                'pred_answer_type': outputs['pred_answer_type'][i:i+1],
            }
            
            qa_logits = {
                'pred_answer_obj': instance_outputs['pred_answer_obj'].cpu(),
                'pred_answer_global': instance_outputs['pred_answer_global'].cpu(),
                'pred_answer_type': instance_outputs['pred_answer_type'].cpu(),
            }
            
            result = {
                'qa_logits': qa_logits,
                'pred_logits': outputs['pred_logits'][i:i+1].cpu(),
                'pred_boxes': outputs['pred_boxes'][i:i+1].cpu(),
            }
            
            # Interpret answer for this instance
            # answer_str, answer_idx, answer_type = self._interpret_qa_output(instance_outputs)
            # result['pred_answer_str'] = answer_str
            # result['pred_answer_idx'] = answer_idx
            # result['pred_answer_type'] = answer_type
            
            results.append(instance_outputs)
        
        return results

    def getlogitsize(self):
        """
        Get the size of the combined QA logit vector.
        
        Returns
        -------
        int
            Size of logit vector (111 for pred_answer_global).
        """
        return 114  # pred_answer_global has 111 classes

    def getlogit(self, resultobj):
        """
        Get the pre-softmax logits from a result object.
        
        For GridQA, we return the global (count) logits by default,
        as most questions are counting questions.
        
        Parameters
        ----------
        resultobj : dict
            Result from forward().
        
        Returns
        -------
        torch.Tensor
            Logit vector.
        """
        qa_logits = resultobj['qa_logits']
        
        # Determine which head to use based on predicted type
        if resultobj['pred_answer_type'] == 'count':
            return qa_logits['pred_answer_global'].detach().squeeze()
        else:
            return qa_logits['pred_answer_obj'].detach().squeeze()

    def getpredlabel(self, resultobj):
        """
        Get the predicted label from a result object.
        
        Parameters
        ----------
        resultobj : dict
            Result from forward().
        
        Returns
        -------
        int
            Predicted answer index.
        """
        return resultobj['pred_answer_idx']

    def getprelinear(self, resultobj):
        """
        Get the pre-linear layer features.
        
        Parameters
        ----------
        resultobj : dict
            Result from forward().
        
        Returns
        -------
        np.ndarray or None
            Pre-linear features if available.
        """
        # MDETR doesn't easily expose pre-linear features
        return None

    def getprelinearsize(self):
        """Get size of pre-linear features."""
        return 0  # Not implemented for MDETR

    def replaceunimodaldata(self, datainstance, modality, newinput):
        """
        Replace data in one modality with new input.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Original data instance.
        modality : str
            'image' or 'text'
        newinput : np.ndarray or str
            New data for the modality.
        
        Returns
        -------
        tuple or dict
            New data instance with replaced modality.
        """
        if isinstance(datainstance, dict):
            new_instance = copy.deepcopy(datainstance)
            if modality == "image":
                if isinstance(newinput, str):
                    new_instance['image'] = np.asarray(Image.open(newinput).convert("RGB"))
                else:
                    new_instance['image'] = newinput
            elif modality == "text":
                new_instance['question'] = newinput
            else:
                raise ValueError(f"Unknown modality: {modality}")
            return new_instance
        else:
            # Tuple format: (image, question, answer, label)
            if modality == "image":
                if isinstance(newinput, str):
                    # Already a file path
                    return (newinput, datainstance[1], datainstance[2], datainstance[3])
                else:
                    # Save numpy array to temp file
                    randname = f"tmp/gridqa_{random.randint(0, 100000000)}.png"
                    os.makedirs("tmp", exist_ok=True)
                    Image.fromarray(newinput.astype(np.uint8)).save(randname)
                    return (randname, datainstance[1], datainstance[2], datainstance[3])
            elif modality == "text":
                return (datainstance[0], newinput, datainstance[2], datainstance[3])
            else:
                raise ValueError(f"Unknown modality: {modality}")

    def _get_normed_image(self, image):
        """
        Normalize image for model input.
        
        Parameters
        ----------
        image : PIL.Image, np.ndarray, or str
            Input image.
        
        Returns
        -------
        torch.Tensor
            Normalized image tensor.
        """
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return self.transform(pil_image)

    def _build_probas(self, outputs):
        """
        Build combined probability tensor from QA head outputs.
        
        For GridQA (GQA-style), we concatenate:
        - pred_answer_global (111 classes for counts/answers)
        - pred_answer_obj (3 classes for obj-level answers)
        
        Parameters
        ----------
        outputs : dict
            Model output dictionary.
        
        Returns
        -------
        torch.Tensor
            Combined logits tensor.
        """
        # pred_answer_global has shape (B, 111)
        # pred_answer_obj has shape (B, 3)
        probas = torch.cat(
            (
                outputs["pred_answer_global"],
                outputs["pred_answer_obj"],
            ),
            dim=1,
        )
        return probas

    def getgrad(self, datainstance, target, prelinear=False):
        """
        Compute gradients with respect to input image.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance.
        target : int
            Target class index for gradient computation.
        prelinear : bool
            If True, compute gradient w.r.t. pre-linear features.
        
        Returns
        -------
        tuple
            (normed_image, grad, imgfile) - normalized image, gradient, and image path/array.
        """
        self.model.zero_grad()
        
        # Extract image and question
        if isinstance(datainstance, dict):
            image = datainstance['image']
            question = datainstance['question']
            imgfile = image  # numpy array
        else:
            image = datainstance[0]
            question = datainstance[1]
            imgfile = image
        
        # Get normalized image with gradient tracking
        normed_image = self._get_normed_image(image).to(self.device)
        normed_image.requires_grad = True
        
        samples = torch.unsqueeze(normed_image, 0).to(self.device)
        captions = [question]
        
        if prelinear:
            model_features = []
            
            def hook(module, input, output):
                nonlocal model_features
                model_feat = input
                model_features.append(model_feat[0][0])
            
            handle1 = self.model.answer_type_head.register_forward_hook(hook)
            handle2 = self.model.answer_obj_head.register_forward_hook(hook)
            handle3 = self.model.answer_global_head.register_forward_hook(hook)
        
        # Forward pass
        memory_cache = self.model(samples, captions, encode_and_save=True)
        outputs = self.model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
        
        # Build combined probas
        probas = self._build_probas(outputs)
        
        if prelinear:
            feats = torch.cat(model_features)
            feats[target].backward()
        else:
            probas[0][target].backward()
        
        grad = normed_image.grad.detach()
        
        if prelinear:
            handle1.remove()
            handle2.remove()
            handle3.remove()
        
        return normed_image, grad, imgfile

    def getgradtext(self, datainstance, target, alltarget=False, prelinear=False):
        """
        Compute gradients with respect to text embeddings.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance.
        target : int
            Target class index.
        alltarget : bool
            If True, compute gradient w.r.t. sum of all logits.
        prelinear : bool
            If True, compute gradient w.r.t. pre-linear features.
        
        Returns
        -------
        tuple
            (res, parsed_words, normed_image, text_ids)
            - res: gradient magnitude for each token
            - parsed_words: list of parsed words
            - normed_image: normalized image tensor
            - text_ids: token IDs
        """
        self.model.zero_grad()
        
        # Extract image and question
        if isinstance(datainstance, dict):
            image = datainstance['image']
            question = datainstance['question']
        else:
            image = datainstance[0]
            question = datainstance[1]
        
        # Get normalized image
        normed_image = self._get_normed_image(image).to(self.device)
        normed_image.requires_grad = True
        
        samples = torch.unsqueeze(normed_image, 0).to(self.device)
        captions = [question]
        
        text_embedding = None
        text_ids = None
        gradd = None
        
        def hook_forward(module, input, output):
            nonlocal text_embedding, text_ids
            text_embedding = output[0]
            text_ids = input[0]
        
        def hook_backward(module, input, output):
            nonlocal gradd
            gradd = output[0][0]
        
        # Register hooks on text encoder embeddings
        handle = self.model.transformer.text_encoder.embeddings.word_embeddings.register_forward_hook(
            hook_forward
        )
        handle22 = self.model.transformer.text_encoder.embeddings.word_embeddings.register_full_backward_hook(
            hook_backward
        )
        
        if prelinear:
            model_features = []
            
            def hook(module, input, output):
                nonlocal model_features
                model_feat = input
                model_features.append(model_feat[0][0])
            
            handle1 = self.model.answer_type_head.register_forward_hook(hook)
            handle2 = self.model.answer_obj_head.register_forward_hook(hook)
            handle3 = self.model.answer_global_head.register_forward_hook(hook)
        
        # Forward pass
        memory_cache = self.model(samples, captions, encode_and_save=True)
        outputs = self.model(samples, captions, encode_and_save=False, memory_cache=memory_cache)
        
        # Build combined probas
        probas = self._build_probas(outputs)
        
        # Free memory_cache and outputs early — only probas is needed for backward
        del memory_cache, outputs, samples
        
        if alltarget:
            torch.sum(probas[0]).backward(create_graph=True)
        elif prelinear:
            feats = torch.cat(model_features)
            feats[target].backward()
        else:
            probas[0][target].backward()
        
        handle.remove()
        handle22.remove()
        
        # Free probas after backward — it holds references to the full graph
        del probas
        
        # Compute token importance as dot product of embedding and gradient
        res = torch.sum(text_embedding * gradd, dim=1)
        
        if prelinear:
            handle1.remove()
            handle2.remove()
            handle3.remove()
        
        return res, self._parse_question(question), normed_image, text_ids

    def _parse_question(self, question):
        """
        Parse question into word tokens.
        
        Parameters
        ----------
        question : str
            Input question.
        
        Returns
        -------
        list
            List of word tokens with special markers.
        """
        words = []
        # Remove trailing punctuation for splitting
        q = question.rstrip('?').strip()
        for word in q.split(" "):
            if word:
                words.append(word)
        words.append("?")
        words.append("<end>")
        words.insert(0, "<start>")
        return words

    def getdoublegrad(self, datainstance, target, targetwords, alltarget=True):
        """
        Compute double gradient (gradient of text gradient w.r.t. image).
        
        This shows which image regions are important for understanding
        specific words in the question.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance.
        target : int
            Target class index.
        targetwords : list
            List of token indices to focus on.
        alltarget : bool
            If True, compute gradient w.r.t. sum of all logits.
        
        Returns
        -------
        tuple
            (grad, parsed_words, text_ids)
            - grad: gradient tensor w.r.t. image
            - parsed_words: list of parsed words
            - text_ids: token IDs
        """
        res, di, normed_image, text_ids = self.getgradtext(
            datainstance, target, alltarget=alltarget
        )
        text_ids_out = text_ids.detach().cpu() if isinstance(text_ids, torch.Tensor) else text_ids
        
        # Accumulate gradients for target words
        ac = 0.0
        if(targetwords is None):
            targetwords = range(len(res))
        for id in targetwords:
            ac += torch.abs(res[id])
        
        # Compute gradient w.r.t. image
        rets = torch.autograd.grad(ac, normed_image)

        # Detach to break reference to computation graph (prevents OOM in loops)
        grad = rets[0].detach().cpu()

        # Aggressively free the computation graph retained by create_graph=True.
        # The .backward(create_graph=True) in getgradtext stores graph-linked
        # .grad tensors on every model parameter; these must be cleared here
        # before returning, otherwise they accumulate across loop iterations.
        del rets, ac, res, normed_image, text_ids
        self.model.zero_grad(set_to_none=True)
        for p in self.model.parameters():
            p.grad = None
        torch.cuda.empty_cache()
                
        return grad, di, text_ids_out

    def get_attention_maps(self, datainstance):
        """
        Extract attention maps from the model.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance.
        
        Returns
        -------
        dict
            Dictionary containing various attention maps.
        """
        # Extract image and question
        if isinstance(datainstance, dict):
            image = datainstance['image']
            question = datainstance['question']
        else:
            image = datainstance[0]
            question = datainstance[1]
        
        img_tensor = self._preprocess_image(image)
        captions = [question]
        
        with torch.no_grad():
            memory_cache = self.model(img_tensor, captions, encode_and_save=True)
            outputs = self.model(img_tensor, captions, encode_and_save=False,
                                memory_cache=memory_cache)
        
        # Extract available attention information
        attention_info = {}
        
        if 'proj_queries' in outputs:
            attention_info['proj_queries'] = outputs['proj_queries'].cpu().numpy()
        if 'proj_tokens' in outputs:
            attention_info['proj_tokens'] = outputs['proj_tokens'].cpu().numpy()
        
        return attention_info

    def get_detection_boxes(self, datainstance, threshold=0.5):
        """
        Get detected object bounding boxes.
        
        Parameters
        ----------
        datainstance : tuple or dict
            Data instance.
        threshold : float
            Confidence threshold for detection.
        
        Returns
        -------
        tuple
            (boxes, scores) - numpy arrays of boxes and confidence scores.
        """
        result = self.forward(datainstance)
        
        # Get objectness scores (1 - background probability)
        pred_logits = result['pred_logits']
        objectness = 1 - F.softmax(pred_logits, dim=-1)[0, :, -1]
        
        # Filter by threshold
        keep = objectness > threshold
        boxes = result['pred_boxes'][0, keep].numpy()
        scores = objectness[keep].numpy()
        
        # Convert from cxcywh to xyxy format
        if len(boxes) > 0:
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack([
                cx - w/2, cy - h/2,
                cx + w/2, cy + h/2
            ], axis=1)
        
        return boxes, scores

    def classnames(self):
        """Return answer class names."""
        return self.answermapping


# Convenience function
def get_gridqa_mdetr(device="cuda", checkpoint_path=None, **kwargs):
    """
    Factory function to create GridQAMDETR instance.
    
    Parameters
    ----------
    device : str
        Device to use.
    checkpoint_path : str, optional
        Path to checkpoint.
    **kwargs
        Additional arguments.
    
    Returns
    -------
    GridQAMDETR
    """
    return GridQAMDETR(device=device, checkpoint_path=checkpoint_path, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing GridQAMDETR...")
    
    # Import dataset
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.datasets.gridvqa.datasets.gridqa import GridQADataset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model with checkpoint
    checkpoint_path = os.environ.get("GRIDVQA_CHECKPOINT", "/path/to/checkpoint.pth")
    print(f"Building model with checkpoint: {checkpoint_path}")
    try:
        model = GridQAMDETR(device=device, checkpoint_path=checkpoint_path)
        print("Model built successfully!")
        
        print(f"Modality names: {model.getmodalitynames()}")
        print(f"Modality types: {model.getmodalitytypes()}")
        print(f"Logit size: {model.getlogitsize()}")
        print(f"Class names (first 10): {model.classnames()[:10]}")
        
        # Test with dataset
        print("\nLoading dataset...")
        dataset = GridQADataset(split="val")
        
        if dataset.length() > 0:
            print("Testing forward pass...")
            sample = dataset.getdata_full(0)
            print(f"  Question: {sample['question']}")
            print(f"  Ground truth answer: {sample['answer']}")
            
            result = model.forward(sample)
            print(f"  Predicted answer: {result['pred_answer_str']}")
            print(f"  Answer type: {result['pred_answer_type']}")
            print("Forward pass successful!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()