import torch
import torch.nn as nn
from einops import rearrange

from vggt.models.tokenizer_blocks import TiTokEncoder, TiTokDecoder
from vggt.models.tokenizer_quantizer import VectorQuantizer, DiagonalGaussianDistribution
import json
from omegaconf import OmegaConf
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin


import os
from typing import Union, Callable, Dict, Optional

import torch


class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def save_pretrained_weight(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Saves a model and its configuration file to a directory.

        Args:
            save_directory: A string or os.PathLike, directory to which to save. 
                Will be created if it doesn't exist.
            save_function: A Callable function, the function to use to save the state dictionary.
                Useful on distributed training like TPUs when one need to replace `torch.save` by
                another method. Can be configured with the environment variable `DIFFUSERS_SAVE_MODE`.
            state_dict: A dictionary from str to torch.Tensor, the state dictionary to save.
                If `None`, the model's state dictionary will be saved.
        """
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        if state_dict is None:
            state_dict = model_to_save.state_dict()
        weights_name = "pytorch_model.bin"

        save_function(state_dict, os.path.join(save_directory, weights_name))

        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def load_pretrained_weight(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        r"""Instantiates a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        Args:
            pretrained_model_path: A string or os.PathLike, a path to a *directory* or *file* containing model weights.

        Raises:
            ValueError: If pretrained_model_path does not exist.
        """
        # If pretrained_model_path is a file, set model_file to this file.
        if os.path.isfile(pretrained_model_path):
            model_file = pretrained_model_path
        # If pretrained_model_path is a directory, set model_file to the path of the 
        # file "pytorch_model.bin" in this directory.
        elif os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.isfile(pretrained_model_path):
                model_file = pretrained_model_path
            else:
                raise ValueError(f"{pretrained_model_path} does not exist")
        else:
            raise ValueError(f"{pretrained_model_path} does not exist")

        # Load model state from checkpoint.
        checkpoint = torch.load(model_file, map_location="cpu")
        # Load state dictionary into self.
        msg = self.load_state_dict(checkpoint, strict=strict_loading)
        # Print information about loading weights.
        print(f"loading weight from {model_file}, msg: {msg}")
        # If torch_dtype is specified and is a valid torch.dtype, convert self to this dtype.
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default.
        self.eval()

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """Gets the number of parameters in the module.

        Args:
            only_trainable: A boolean, whether to only include trainable parameters.
            exclude_embeddings: A boolean, whether to exclude parameters associated with embeddings.

        Returns:
            An integer, the number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)


class TiTok(BaseModel, PyTorchModelHubMixin, tags=["arxiv:2406.07550", "image-tokenization"], repo_url="https://github.com/bytedance/1d-tokenizer", license="apache-2.0"):
    def __init__(self, 
                image_size, 
                patch_size, 
                num_register_tokens,
                quantize_mode="vq", 
                model_size="large", 
                num_latent_tokens=32, 
                token_size=12,
                codebook_size=4096,
                commitment_cost=0.25,
                use_l2_norm=True
        ):

        super().__init__()
        self.quantize_mode = quantize_mode
        if self.quantize_mode not in ["vq", "vae"]:
            raise ValueError(f"Unsupported quantize mode {self.quantize_mode}.")

        assert num_register_tokens >= 0

        self.encoder = TiTokEncoder(
            image_size=image_size,
            patch_size=patch_size,
            model_size=model_size,
            num_latent_tokens=num_latent_tokens,
            num_register_tokens=num_register_tokens,
            token_size=token_size,
            quantize_mode=quantize_mode)

        
        self.num_latent_tokens = num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        # if self.quantize_mode == "vq":
        #     self.quantize = VectorQuantizer(
        #         codebook_size=codebook_size,
        #         token_size=token_size,
        #         commitment_cost=commitment_cost,
        #         use_l2_norm=use_l2_norm)
        # elif self.quantize_mode == "vae":
        #     self.quantize = DiagonalGaussianDistribution
        # else:
        #     raise NotImplementedError

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        # if self.quantize_mode == "vq":
        #     z_quantized, result_dict = self.quantize(z)
        # elif self.quantize_mode == "vae":
        #     posteriors = self.quantize(z)
        #     z_quantized = posteriors.sample()
        #     result_dict = posteriors

        return z
    
    def forward(self, x):
        return self.encode(x)