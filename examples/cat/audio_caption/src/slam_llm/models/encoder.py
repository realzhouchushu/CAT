import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from timm.models.layers import to_2tuple

# audio mae patch embed
class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ASTEncoder:
    @classmethod
    def load(cls, model_config):
        os.environ['TORCH_HOME'] = '~/hubs/models/torchhome'
        from .ast.src.models import ASTModel
        ast_mdl = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128,
                                  input_tdim=1024, imagenet_pretrain=False,
                                  audioset_pretrain=False, model_size='base384')
        checkpoint = torch.load(model_config.encoder_path, map_location='cuda')
        ast_mdl = torch.nn.DataParallel(ast_mdl, device_ids=[0])
        ast_mdl.load_state_dict(checkpoint)
        ast_mdl = ast_mdl.module
        return ast_mdl

class AudioMAEEncoder:
    @classmethod
    def load(cls, model_config):
        from .AudioMAE import models_vit
        args_model = 'vit_base_patch16'
        args_use_custom_patch = False
        model = models_vit.__dict__[args_model](
            num_classes=527,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=True,
            use_custom_patch=args_use_custom_patch,
            ## remove video part for A-MAE
            #load_video=args.load_video,
            # n_frm=args.n_frm,
            #split_pos=args.split_pos,
            #av_fusion=args.av_fusion,
        )
        img_size = (1024, 128)
        in_chans = 1
        emb_dim = 768
        if args_model == "vit_small_patch16":
            emb_dim = 384
        if args_use_custom_patch:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim, stride=10)
            model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        else:
            model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
            num_patches = model.patch_embed.num_patches
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
        
        checkpoint = torch.load(model_config.encoder_path, weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        model.load_state_dict(checkpoint_model, strict=False)

        # from .BEATs.BEATs import BEATs, BEATsConfig
        # checkpoint = torch.load(model_config.encoder_path)
        # cfg = BEATsConfig(checkpoint['cfg'])
        # BEATs_model = BEATs(cfg)
        # BEATs_model.load_state_dict(checkpoint['model'])

        return model

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        if model_config.whisper_decode:
            import whisper
            whisper_model = whisper.load_model(name=model_config.encoder_path, device='cpu')
            whisper_model.encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, whisper_model.encoder)
            return whisper_model

        if model_config.encoder_path_hf is not None:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(model_config.encoder_path_hf,torch_dtype=torch.bfloat16).encoder
        else:
            import whisper
            encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
            encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder


class BEATsEncoder:

    @classmethod
    def load(cls, model_config):
        from .BEATs.BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])

        return BEATs_model


@dataclass
class UserDirModule:
    user_dir: str
    
class EATEncoder:
    
    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        EATEncoder = EATEncoder[0]

        return EATEncoder
    
    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask = padding_mask, mask=False, remove_extra_tokens = False)['x']

class CLAPEncoder: 

    @classmethod
    def load(cls, model_config): 
        from .CLAP.ase_model import ASE
        import ruamel.yaml as yaml
        with open(model_config.clap_config, 'r') as f: 
            clap_config = yaml.safe_load(f)
        clap_config['pd_text_support'] = model_config.get("pd_text_support", None)
        model = ASE(clap_config)
        checkpoint = torch.load(model_config.encoder_path)['model']
        model.load_state_dict(checkpoint)
        return model
    
class SpatialASTEncoder:
    @classmethod
    def load(cls, model_config):
        from functools import partial
        from .SpatialAST import SpatialAST 
        binaural_encoder = SpatialAST.BinauralEncoder(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        checkpoint = torch.load(model_config.encoder_ckpt, map_location='cpu')
        binaural_encoder.load_state_dict(checkpoint['model'], strict=False) 
        return binaural_encoder

class WavLMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .wavlm.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"
 
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        return model

class HubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model


class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model

class MusicFMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .musicfm.model.musicfm_25hz import MusicFM25Hz
        model = MusicFM25Hz(
            stat_path = model_config.encoder_stat_path,
            model_path = model_config.encoder_path,
            w2v2_config_path = model_config.get('encoder_config_path', "facebook/wav2vec2-conformer-rope-large-960h-ft")
        )
        return cls(model_config, model)

    def extract_features(self, source, padding_mask=None):
        _, hidden_states = self.model.get_predictions(source)
        out = hidden_states[self.config.encoder_layer_idx]
        return out

class Emotion2vecEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = model[0]

        return model