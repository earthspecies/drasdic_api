import yaml
import torch

def get_model(args, device = None):
        
    if args['model_type'] == "encoder_decoder":
        from drasdic.models.encoder_decoder import EncoderDecoder
        model = EncoderDecoder(args)
    elif args['model_type'] == 'framewise_protonet':
        from drasdic.models.encoder_decoder import FramewiseProtonet
        model = FramewiseProtonet(args)
    elif args['model_type'] == 'bled':
        from drasdic.models.bled import BLED
        model = BLED(args)
    else:
        raise ValueError

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
    if args['previous_checkpoint_fp'] is not None:
        print(f"loading model weights from {args['previous_checkpoint_fp']}")
        cp = torch.load(args['previous_checkpoint_fp'], map_location = device)
        missing_keys, unexpected_keys = model.load_state_dict(cp["model_state_dict"], strict=False)
        
    if args['previous_encoder_fp'] is not None:
        print(f"loading model weights from {args['previous_encoder_fp']}")
        state_dict = torch.load(args['previous_encoder_fp'])
        model.encoder.load_state_dict(state_dict)
        
    return model
        