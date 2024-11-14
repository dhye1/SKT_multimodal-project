# part of the code was referenced from SUPERB: https://github.com/s3prl/s3prl
# and https://github.com/wngh1187/IPET/blob/main/Speechcommands_V2/W2V2/models/W2V2.py
import os
import pdb
import copy
import torch
import argparse
import numpy as np
import loralib as lora 
import transformers.models.whisper.modeling_whisper as whisper

from functools import lru_cache

from torch import nn
from adapter import Adapter
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import WavLMModel, WhisperModel, AutoFeatureExtractor

'''
NOTE: 
multimodal_cocat을 위한 whisper 

AutoFeatureExtractor load FeatureExtractor from pre-trained model(e.g., whisperfeaturextractor for openai/whisper-tiny) 
whisperfeaturextractor transform raw signal to mel-filter bank feature  

LoRA - FFN, Atten_key, Atten_q, Atten_v 까지 구현된 파일. 
weighted sum 추가 
WG 추가 - clamp추가 (0.2-0.9사이로 조정)

self.weights = nn.Parameter(torch.ones(num_layers)) 로 수정 

TODO:
Atten output에 LoRA, - output projection ...  
WG adapter초기화 어떻게 할지 ... 

Cross modal 합힐때 주의 사항: 
- Cross modal attention -> 평균 취하지 않고 그대로 (batch, seq, hidden_state) 할 수 있도록 
'''

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 80):
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        ## LoRA_attn
        if self.config.finetune_method == "lora_attn" \
        or self.config.finetune_method == "lora_all" \
        or self.config.finetune_method == "all":
            if self.config.is_key_lora:
                self.q_proj = lora.Linear(config.d_model, config.d_model, r=config.lora_rank)
                self.k_proj = lora.Linear(config.d_model, config.d_model, r=config.lora_rank)
                self.v_proj = lora.Linear(config.d_model, config.d_model, r=config.lora_rank)
            else:
                self.q_proj = lora.Linear(config.d_model, config.d_model, r=config.lora_rank)
                self.v_proj = lora.Linear(config.d_model, config.d_model, r=config.lora_rank)

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value= None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value

class WhisperEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config = config 
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.config = config
        
        if self.config.finetune_method == "lora" or self.config.finetune_method == "lora_all" or self.config.finetune_method == "all":
            self.fc1 = lora.Linear(self.embed_dim, config.encoder_ffn_dim, r=config.lora_rank)
            self.fc2 = lora.Linear(config.encoder_ffn_dim, self.embed_dim, r=config.lora_rank)

            
        if self.config.finetune_method == "adapter" \
        or self.config.finetune_method == "adapter_l"  \
        or self.config.finetune_method == "all":
            self.adapter = Adapter(
                config, 
                d_model=self.embed_dim,
                dropout=0.1, 
                bottleneck=config.adapter_hidden_dim, 
                adapter_scalar=0.1
            )
        
        # weight gating params 
        if self.config.wg:
            self.wg = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.wg, mean=0.0, std=0.02)
            # self.sig = nn.Sigmoid() # ones로 init 
            # nn.init.kaiming_uniform_(self.wg) # xavier_uniform_

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        
        if self.config.wg: 
            # wg_scale = torch.sigmoid(self.wg)
            # print(f"BEFORE hidden_states: {hidden_states}")
            
            # wg_scale = torch.tanh(self.wg)
            wg_scale = torch.sigmoid(self.wg)
            # wg_scale = torch.clamp(wg_scale, min=0.2, max=0.9) 
            hidden_states = wg_scale * hidden_states

            # print(f"wg_scale: {wg_scale}")
            # print(f"AFTER hidden_states: {hidden_states}")
            # hidden_states = self.sig(self.wg) * hidden_states 
        
        # print("attention_mask: ", attention_mask)
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        # Adapter
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "all":
            adapt_h = self.adapter(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Adapter
        if self.config.finetune_method == "adapter_l": 
        # or self.config.finetune_method == "all": 
            hidden_states = hidden_states + self.adapter(hidden_states)
        
        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        
        # Adapter
        if self.config.finetune_method == "adapter" or self.config.finetune_method == "all": 
            hidden_states = hidden_states + adapt_h
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
   
class WhisperWrapper(nn.Module):
    def __init__(
        self, 
        args, 
        hidden_dim=256,
        output_class_num=4
        # rep = 'encoded' # 'downstream'
    ):
        super(WhisperWrapper, self).__init__()
        # 1. We Load the model first with weights
        self.args = args
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
        if args.audio_model == "whisper_tiny":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-tiny",
                output_hidden_states=True
            )
        elif args.audio_model == "whisper_base":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-base",
                output_hidden_states=True
            )
        elif args.audio_model == "whisper_small":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-small",
                output_hidden_states=True
            )
        elif args.audio_model == "whisper-medium":
            self.backbone_model = WhisperModel.from_pretrained(
                "openai/whisper-medium",
                output_hidden_states=True
            )
        
        self.embed_positions = copy.deepcopy(self.backbone_model.encoder.embed_positions.weight)
        self.embed_positions.requires_grad = False
        state_dict = self.backbone_model.state_dict()
        
        # 2. Read the model config
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method        = args.finetune_method
        self.model_config.adapter_hidden_dim     = args.adapter_hidden_dim
        self.model_config.lora_rank              = args.lora_rank
        self.model_config.is_key_lora            = args.is_key_lora
        self.model_config.wg                     = args.wg
        
        
        # 3. Config encoder layers with adapter or embedding prompt
        self.backbone_model.encoder.layers = nn.ModuleList(
            [WhisperEncoderLayer(self.model_config) for _ in range(self.model_config.encoder_layers)]
        )
        # 4. Load the weights back
        msg = self.backbone_model.load_state_dict(state_dict, strict=False)
        # 5. Freeze the weights
        if self.args.finetune_method == "adapter" or self.args.finetune_method == "adapter_l" \
        or self.args.finetune_method == "finetune" \
        or self.args.finetune_method == "lora" or self.args.finetune_method == "lora_attn" or self.args.finetune_method == "lora_all" \
        or self.args.finetune_method == "all" or self.args.ws == True:
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False
        else: ## whisper frozen 
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False

        self.finetune_method = self.args.finetune_method
        self.downstream = args.downstream 
        self.ws = args.ws
        
        # 6. Downstream models
        if self.downstream:
            
            self.model_seq = nn.Sequential(
                nn.Conv1d(self.model_config.hidden_size, hidden_dim, 1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
            )
            self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
            )
        
        if args.use_conv_output:
            num_layers = self.model_config.num_hidden_layers + 1  # transformer layers + input embeddings
            self.weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        else:
            num_layers = self.model_config.num_hidden_layers
            self.weights = nn.Parameter(torch.zeros(num_layers))
            # weights = torch.zeros(num_layers)
            # #weights[-1] = 1.0
            # self.weights = nn.Parameter(weights) #
            # nn.init.normal_(self.weights, mean=0.0, std=0.02) 

        if self.downstream:
            self.out_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_class_num),
            )
        
    def forward(self, x, length=6):
        # 1. feature extraction and projections
        if length is not None:
            max_audio_len = length.max().detach().cpu()
            # Append to list for feature_extractor to work
            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())
            
            # Max length is max audio len in a batch
            features = self.feature_extractor(
                new_x,
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=max_audio_len
            )
            features = features.input_features.cuda()
        else:
            features = self.feature_extractor(
                x[0].detach().cpu(), 
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=len(x[0])
            )
            features = features.input_features.cuda()
            # print("whisper feature", features.shape)
        # 2. get length and mask
        if length is not None:
            length = self.get_feat_extract_output_lengths(length.detach().cpu())
            max_len = length.max()
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:max_len])
        else:
            tmp_length = self.get_feat_extract_output_lengths(len(x[0]))
            # Replace positional embeddings
            self.backbone_model.encoder.embed_positions = self.backbone_model.encoder.embed_positions.from_pretrained(self.embed_positions[:tmp_length])
            
        # 3. transformer encoding features
        
        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        )
        
        encoded_feature = features.last_hidden_state

        if self.downstream == False:
            # print("downstream false")
            # [Weighted sum hidden states]  
            if self.ws:
                # print("weghited sum ")
                features = features.hidden_states
            
                # 4. stacked feature
                if self.args.use_conv_output:
                    stacked_feature = torch.stack(features, dim=0)
                else: # defualt; 
                    stacked_feature = torch.stack(features, dim=0)[1:]
                    # print(f"stacked_features {stacked_feature.shape}")
                
                # 5. Weighted sum
                _, *origin_shape = stacked_feature.shape
                # Return transformer enc outputs [num_enc_layers, B, T, D]
                if self.args.use_conv_output:
                    stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
                else: # defualt;  
                    stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
                    # print(f"transformer enc outputs {stacked_feature.shape}")
                norm_weights = F.softmax(self.weights, dim=-1)
                
                # Perform weighted average
                weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
                encoded_feature = weighted_feature.view(*origin_shape) 
            
            # [Use only last_hidden_state]
            # else:
            #     # [batch, 300, encoder_dim]
            #     encoded_feature = features.last_hidden_state 

            
            # Cross Modal Attention 
            if self.args.modal == 'multimodal' and self.args.cross_modal_atten: 
                # print("multimodal and cross_modal_atten")
                # print("encoded_feature.shape", encoded_feature.shape)
                if length is not None: # [batch, encoder_dim]
                    length = length.cuda()
                    masks = torch.arange(encoded_feature.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
                    masks = masks.float()
                    encoded_feature = (encoded_feature * masks.unsqueeze(-1))#.sum(1) / length.unsqueeze(1)

                    # print("mask", masks)
                    # print("mask.shape", masks.shape)
                    # print("encoded_feature.shape", encoded_feature.shape)
    
                return (encoded_feature, masks) # masks is used on cross modal attention 

            # Cross Modal Attention + Additional Whisper feature 
            elif self.args.modal == 'multimodal_concat' and self.args.cross_modal_atten: 
                print("multimodal_concat")
                if length is not None: # [batch, encoder_dim]
                    length = length.cuda()
                    masks = torch.arange(encoded_feature.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
                    masks = masks.float()
                    encoded_feature = (encoded_feature * masks.unsqueeze(-1))
                    encoded_feature_audio = (encoded_feature * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
                    return (encoded_feature, masks, encoded_feature_audio)
            
            # Unimodal or Concat 
            else:
                if length is not None: # [batch, encoder_dim]
                    print("Unimodal or Concat / length is not None ")
                    length = length.cuda()
                    masks = torch.arange(encoded_feature.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
                    masks = masks.float()
                    encoded_feature = (encoded_feature * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
                    # batch 내부에 max len에 맞춰서 
                    # print("mask", masks)
                    # print("mask.shape", masks.shape)
                    # print("length", length)
                 
                else:
                    print("Unimodal or Concat / length is None ")
                    encoded_feature = torch.mean(encoded_feature, dim=1)
                return encoded_feature
        
        elif self.downstream: # args.downstream

            print("downstream")
            features = features.hidden_states
           
            # 4. stacked feature
            if self.args.use_conv_output:
                stacked_feature = torch.stack(features, dim=0)
            else: # defualt; 
                stacked_feature = torch.stack(features, dim=0)[1:]
                # print(f"stacked_features {stacked_feature.shape}")
            
            # 5. Weighted sum
            _, *origin_shape = stacked_feature.shape
            # Return transformer enc outputs [num_enc_layers, B, T, D]
            if self.args.use_conv_output:
                stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers+1, -1)
            else: # defualt;  
                stacked_feature = stacked_feature.view(self.backbone_model.config.num_hidden_layers, -1)
                # print(f"transformer enc outputs {stacked_feature.shape}")
            norm_weights = F.softmax(self.weights, dim=-1)
            
            # Perform weighted average
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            features = weighted_feature.view(*origin_shape) ### weghited sum 
        
            
            # 6. Pass the weighted average to point-wise 1D Conv
            # B x T x D [32, 300, 256]
            features = features.transpose(1, 2)
            features = self.model_seq(features)
            features = features.transpose(1, 2)
            #print(f"point-wise 1D Conv {features.shape}") 
            
            # 7. Pooling [32, 256]
            if length is not None:
                length = length.cuda()
                masks = torch.arange(features.size(1)).expand(length.size(0), -1).cuda() < length.unsqueeze(1)
                masks = masks.float()
                features = (features * masks.unsqueeze(-1)).sum(1) / length.unsqueeze(1)
            else:
                features = torch.mean(features, dim=1)
            
            # 8. Output predictions
            # B x D
            predicted = self.out_layer(features)
            # return predicted
            return features
        
        
    # From huggingface
    def get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

def prepare_mask(length, shape, dtype):
    # Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[(torch.arange(mask.shape[0]), length.cpu() - 1)] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--pretrained_model', 
        default='whisper_tiny',
        type=str, 
        help='finetune method: whisper_tiny, whisper_base, whisper_small'
    )
    
    parser.add_argument(
        '--finetune_method', 
        default='none',
        type=str, 
        help='finetune method: adapter, embedding prompt, input prompt'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim', 
        default=128,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--lora_rank', 
        default=16,
        type=int, 
        help='adapter dimension'
    )
    
    args = parser.parse_args()
    model = WhisperWrapper(args).cuda()
    data = torch.zeros([1, 16000]).cuda()
    output = model(data)
    print(output.shape)