"""
Constructor for model based on input args
"""

from typing import Any, Dict, Optional

import torch
from torch import nn


def get_model(args: Dict[str, Any], device: Optional[str] = None) -> nn.Module:
    """
    Construct and return a model based on the specified arguments.

    Parameters
    ----------
    args : dict
        A dictionary of configuration options. Expected keys:
            - "model_type" : str
                The type of model to construct. Must be "encoder_decoder".
            - "previous_checkpoint_fp" : Optional[str]
                Filepath to a checkpoint from which to load the full model state.
            - "previous_encoder_fp" : Optional[str]
                Filepath to a checkpoint from which to load only the encoder weights.
    device : str, optional
        Device to load the model on. If None, will automatically use "cuda" if available,
        otherwise "cpu".

    Returns
    -------
    model : torch.nn.Module
        The constructed model instance.

    Raises
    ------
    ValueError
        If an unsupported model_type is specified.
    """

    if args["model_type"] == "encoder_decoder":
        from drasdic.models.encoder_decoder import EncoderDecoder

        model = EncoderDecoder(args)
    else:
        raise ValueError

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args["previous_checkpoint_fp"] is not None:
        print(f"loading model weights from {args['previous_checkpoint_fp']}")
        cp = torch.load(args["previous_checkpoint_fp"], map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(cp["model_state_dict"], strict=False)

    if args["previous_encoder_fp"] is not None:
        print(f"loading model weights from {args['previous_encoder_fp']}")
        state_dict = torch.load(args["previous_encoder_fp"])
        model.encoder.load_state_dict(state_dict)

    return model
