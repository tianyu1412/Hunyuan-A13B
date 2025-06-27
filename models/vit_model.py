import json
import types
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from contextlib import contextmanager
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,  
    _prepare_4d_causal_attention_mask_for_sdpa, 
    _prepare_4d_causal_attention_mask,
)
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .modeling_hunyuan import HunYuanDecoderLayer, HunYuanRMSNorm
from .configuration_hunyuan import HunYuanConfig


def NaVitForward(input_ids, encoder_input, vit, image_tensors, images_pos, vit_input_resolution, im_start_id, im_end_id, image_token_id, anyres_vit_two_views, dtype):
    # input_ids: (B, L)
    # encoder_input: (L, B, E)
    # image_tensors [[Tensor],...,[Tensor]]
    # image_pos [[Tensor],...,[Tensor]]
    # tokenizer = get_tokenizer()
    b = len(input_ids)
    img_embs = None
    all_nums = sum([len(tensors) for tensors in image_tensors]) if image_tensors else 0
    if all_nums != 0:
        img_embs, img_batch_pos = vit(image_tensors)
    else:
        # when no input image, initialize a fake tensor
        pad_nums = 1
        image_tensors = [[torch.rand(3, vit_input_resolution, vit_input_resolution, dtype=dtype, device=torch.cuda.current_device()) for _ in range(pad_nums)]]
        img_embs, img_batch_pos = vit(image_tensors)

    encoder_input = encoder_input.clone()
    if all_nums > 0:
        assert len(images_pos) == len(img_batch_pos), \
                (len(images_pos), len(img_batch_pos))
        start_token_id = im_start_id
        end_token_id = im_end_id
        placeholder_id = image_token_id
        for idx in range(len(images_pos)):
            assert len(images_pos[idx]) == len(img_batch_pos[idx]), \
                (len(images_pos[idx]), len(img_batch_pos[idx]))
            for p_img_pos_in_batch, p_batch_img_pos in zip(img_batch_pos[idx], images_pos[idx]):
                # the positions to be filled [s_start, s_end)
                s_idx, s_start, s_end = p_img_pos_in_batch
                current_embs = img_embs[s_idx, s_start:s_end]
                im_s, im_e = p_batch_img_pos
                assert len(current_embs) == im_e - im_s, \
                        (img_embs.shape, (s_start, s_end, s_idx), current_embs.shape, (im_s, im_e, idx))
                if not anyres_vit_two_views:
                    assert input_ids[idx, im_s - 1] == start_token_id, \
                            input_ids[idx, im_s - 1]
                    assert input_ids[idx, im_e] == end_token_id, \
                            input_ids[idx, im_e]
                assert (input_ids[idx, im_s:im_e] == placeholder_id).all(), \
                        f'The tokens to be filled are not the placeholder_id {placeholder_id}: {(input_ids[idx, im_s:im_e] == placeholder_id).sum()} vs {im_e - im_s}'
                encoder_input[idx, im_s:im_e] = current_embs
    else:
        # when no input image, to mask vit value
        vit_mask = torch.zeros([1, img_embs.shape[0]], device=torch.cuda.current_device())
        current_embs = img_embs[0, :]
        encoder_input[0, 1:img_embs.shape[0] + 1] = encoder_input[0, 1:img_embs.shape[0] + 1] * (1 - vit_mask) + current_embs * vit_mask
    return encoder_input, input_ids


def VitForward(input_ids, encoder_input, vit, vit_linear_encoder, image_tensors, images_pos, vit_input_resolution, vit_mapping_type, vit_patch, vit_token):
    vit_patch_mlp = (vit_patch > 1 and vit_mapping_type == 'mlp') or vit_patch == 0

    b = len(input_ids)
    if images_pos is None:
        images_pos = torch.ones([len(input_ids), 1, 3])
        images_pos[:, :, 1] = images_pos[:, :, 1]*(vit_token + 1)
        images_pos = images_pos.long()

    real_image_nums = []
    image_tensors = image_tensors.view(b, -1, 3, vit_input_resolution, vit_input_resolution)
    real_images = []

    all_nums = 0
    img_index = []
    for s in range(len(images_pos)):
        real_image_num = 0
        for (im_s, im_e,index) in images_pos[s]:
            if im_s == -1:
                break
            real_image_num += 1
            all_nums += 1
            img_index.append(index)

        real_image_nums.append(real_image_num)
        real_images.append(image_tensors[s][:real_image_num])

    if vit_patch == 1:
        img_index = None

    if all_nums == 0:
        # when no input image, initialize a fake tensor
        img_input = torch.rand(b, 3, vit_input_resolution, vit_input_resolution).cuda().type(image_tensors.dtype)
        img_embs = vit(img_input)
        img_embs = vit_linear_encoder(img_embs)
    else:
        img_input = torch.cat(real_images)
        img_embs = vit(img_input, img_index = img_index)
        img_embs = vit_linear_encoder(img_embs)

    encoder_input = encoder_input.clone()
    start = 0
    if all_nums > 0:
        for s, real_image_len in enumerate(real_image_nums):
            current_embs = img_embs[start:start + real_image_len, :] #[30, 256, 4096]
            for ss in range(current_embs.shape[0]):
                im_s, im_e, index = images_pos[s, ss]
                # 子图特征更少
                if index > 0 and vit_patch_mlp:
                    encoder_input[s, im_s:im_e,] = current_embs[ss, :(im_e-im_s)]
                else:
                    encoder_input[s, im_s:im_e] = current_embs[ss, :]
            start = start + real_image_len
    else:
        # when no input image, to mask vit value
        for s in range(b):
            vit_mask = torch.zeros([vit_token, 1]).cuda()
            current_embs = img_embs[:, start:start + 1]
            encoder_input[1:vit_token + 1, s] = encoder_input[1:vit_token + 1, s] * (1 - vit_mask) + current_embs[:, 0, :] * vit_mask
            start = start + 1
    return encoder_input, input_ids


def group_images_by_max_seq_len(
    images: List[List[Tensor]], patch_size: int, 
    max_seq_len: int, adaptor_patch_size: int, 
    add_cls_token: bool = False) -> List[List[Tensor]]:

    groups = []
    group = []
    pos_groups = []
    seq_len = 0
    num_images = 0
    for image_list in images:
        pos_group = []
        for image in image_list:
            num_images += 1
            assert isinstance(image, Tensor)

            image_dims = image.shape[-2:]
            ph, pw = map(lambda t: t // patch_size, image_dims)

            image_seq_len = (ph * pw)
            new_image_seq_len = image_seq_len
            grouped_len = seq_len + image_seq_len
            if add_cls_token:
                new_image_seq_len += 1
                grouped_len += num_images
                
            assert new_image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'
            
            if grouped_len > max_seq_len:
                groups.append(group)
                group = []
                seq_len = 0
                num_images = 1

            group.append(image)
            start = seq_len // (adaptor_patch_size * adaptor_patch_size)
            end = start + image_seq_len//(adaptor_patch_size * adaptor_patch_size)
            batch_idx = len(groups)
            pos_group.append([batch_idx, start, end])
            seq_len += image_seq_len
        pos_groups.append(pos_group)

    if len(group) > 0:
        groups.append(group)

    return groups, pos_groups


class AnyResCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()

        self.config = config
        # self.sparse_attn_mask = args.sparse_attn_mask
        # self.use_flash_attn = args.use_flash_attn
        self.embed_dim = config.hidden_size
        self.image_size = config.max_image_size
        self.patch_size = config.patch_size
        self.max_seq_len = config.max_vit_seq_len
        self.adaptor_patch_size = config.adaptor_patch_size
        self.anyres_vit_two_views = config.anyres_vit_two_views
        self.vit_add_patchemb_bias = config.vit_add_patchemb_bias
        self.vit_remove_prenorm = config.vit_remove_prenorm

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.vit_add_patchemb_bias,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.skip_cls_token = True

        # add interpolate_pos_encoding
        if self.anyres_vit_two_views:
            self.num_positions = self.num_patches
            self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim) * 0.02)
        else:
            self.num_positions = self.num_patches + 1
            self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
            # self.position_ids = torch.arange(self.num_positions).expand((1, -1))
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        if not self.vit_remove_prenorm:
            self.pre_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1]
        position_embeddings = self.position_embedding(self.position_ids)
        patch_pos_embed = position_embeddings[:, 1:]
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
             return patch_pos_embed
        # class_pos_embed = position_embeddings[:, 0]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        raw_type = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32, non_blocking=True),
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bilinear",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.to(raw_type, non_blocking=True)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def rescale_positional_embedding(self, out_size):
        h, w = out_size
        pos_embed_shape = int((self.position_embedding.shape[1]) ** 0.5)
        if (h, w) == (pos_embed_shape, pos_embed_shape):
            return self.position_embedding
        rescaled_positional_embedding = \
            self.position_embedding.new_zeros(1, h*w, self.position_embedding.shape[2])
        pe_2d = self.position_embedding[0].T.contiguous().view(1, -1, pos_embed_shape, pos_embed_shape)
        pe_2d = F.interpolate(pe_2d, out_size, mode='bilinear', align_corners=False).view(-1, h*w)
        rescaled_positional_embedding[0] = pe_2d.T.contiguous()
        return rescaled_positional_embedding

    def forward_single(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]
        batch_size, num_channels, height, width = pixel_values.shape

        if self.anyres_vit_two_views:
            # padding
            pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))

        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        b, c, h, w = patch_embeds.shape

        # (b, hw, c)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if self.anyres_vit_two_views:
            embeddings = patch_embeds + self.rescale_positional_embedding(out_size=(h, w))
        else:
            embeddings = patch_embeds + self.interpolate_pos_encoding(patch_embeds, height, width)
        if not self.vit_remove_prenorm:
            embeddings = self.pre_layernorm(embeddings)
        return embeddings, (h, w)

    def forward(self, images: List[List[Tensor]]):
        '''
        Input:
            images: List[List[Tensor]]

        Return:
            embeddings: Tensor (B, L, E)
            attn_mask: Tensor (B, L, 2)
            pos_groups: List[List[(batch_idx, start, end)]]
        '''
        batched_images, pos_groups = group_images_by_max_seq_len(
            images, self.patch_size, self.max_seq_len, self.adaptor_patch_size, add_cls_token=not self.skip_cls_token)
        max_seq_len = self.max_seq_len

        # batched_images is a list of a list
        B = len(batched_images)
        L = max_seq_len
        E = self.embed_dim

        embeddings = torch.zeros(B, L, E, dtype=self.config.torch_dtype, requires_grad=True).cuda(non_blocking=True)
        attn_mask = embeddings.new_full((B, 1, L, L), False, dtype=torch.bool)  # True presents compute
        assert len(images) == len(pos_groups), (len(images), len(pos_groups))

        batch_images = []
        batch_pos = []
        for images_i, pos_group in zip(images, pos_groups):
            assert len(images_i) == len(pos_group), (len(images_i), len(pos_group))
            for image, pos in zip(images_i, pos_group):    
                batch_idx, start, end = pos
                a2 = self.adaptor_patch_size ** 2
                # recover the real number of the input image tokens
                start *= a2
                end *= a2
                emb, _ = self.forward_single(image)
                assert emb.ndim == 3, '(B, L, E)'
                embeddings[batch_idx, start:end] = emb
                attn_mask[batch_idx, :, start:end, start:end] = True
        return embeddings, attn_mask, pos_groups


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig, add_pre_layernorm=False, skip_cls_token=True, vit_patch=1):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.image_size = config.vit_input_resolution
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.skip_cls_token = skip_cls_token

        self.num_positions = self.num_patches + 1

        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
        if vit_patch > 1:
            self.position_embedding = nn.Embedding(self.num_patches * (vit_patch ** 2 + 1) + 1, self.embed_dim)
        # 0 支持最大16张图，目前写死了，如需其他的需要额外定义参数
        elif vit_patch == 0:
            self.position_embedding = nn.Embedding(self.num_patches * (16 ** 2 + 1) + 1, self.embed_dim)
        else:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        if add_pre_layernorm:
            self.pre_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        else:
            self.pre_layernorm = None

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        num_patches = embeddings.shape[1] - 1
        position_embeddings = self.position_embedding(self.position_ids)
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
             return position_embeddings
        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        raw_type = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.float(),
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        # print(patch_pos_embed.shape)
        patch_pos_embed = patch_pos_embed.to(raw_type)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding: bool = False, img_index=None) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if self.skip_cls_token:
            embeddings = patch_embeds
            if img_index is None:
                position_ids = self.position_ids[:,1:]
                embeddings = embeddings + self.position_embedding(position_ids)
            else:
                position_ids = (torch.tensor(img_index).cuda() * (self.num_positions - 1)).unsqueeze(1).repeat(1, self.num_positions - 1) \
                    + self.position_ids.expand(batch_size, -1)[:, 1:]
                embeddings = embeddings + self.position_embedding(position_ids)
        else:
            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            if interpolate_pos_encoding:
                embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
            else:
                if img_index is None:
                    embeddings = embeddings + self.position_embedding(self.position_ids)
                else:
                    position_ids = self.position_ids.expand(batch_size,-1)[:,0].unsqueeze(1)
                    new_position = (torch.tensor(img_index).cuda() * (self.num_positions -1)).unsqueeze(1).repeat(1,self.num_positions-1) +  self.position_ids.expand(batch_size,-1)[:,1:]
                    position_ids = torch.cat([position_ids,new_position],dim=1)
                    embeddings = embeddings + self.position_embedding(position_ids)
        if self.pre_layernorm is not None:
            embeddings = self.pre_layernorm(embeddings)
        return embeddings


class NaVitTransformer(nn.Module):
    def __init__(self, config: HunYuanConfig, vit_config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.vit_config = vit_config
        with self.prepare_args(config, vit_config):
            self._use_sdpa = config._attn_implementation == "sdpa"
            self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
            self.layers = nn.ModuleList(
                [HunYuanDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )

    @contextmanager
    def prepare_args(self, config, vit_config):
        hidden_act = config.hidden_act
        hidden_size = config.hidden_size
        ffn_hidden_size = config.intermediate_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        attention_head_dim = config.attention_head_dim
        use_qk_norm = config.use_qk_norm
        use_rotary_pos_emb =  config.use_rotary_pos_emb
        num_hidden_layers = config.num_hidden_layers   
        rms_norm_eps = config.rms_norm_eps
        attention_dropout = config.attention_dropout
        # hidden_dropout = config.hidden_dropout
        norm_type = config.norm_type
        attention_bias = config.attention_bias
        mlp_bias = config.mlp_bias
        use_mla = config.use_mla
        num_experts = config.num_experts
        _attn_implementation = config._attn_implementation
        
        config.hidden_act = vit_config.hidden_act
        config.hidden_size = vit_config.hidden_size
        config.intermediate_size = vit_config.intermediate_size
        config.num_attention_heads = vit_config.num_attention_heads
        config.num_key_value_heads = None
        config.attention_head_dim = vit_config.hidden_size // vit_config.num_attention_heads
        config.use_qk_norm = False
        config.use_rotary_pos_emb = False
        config.num_hidden_layers = vit_config.num_hidden_layers
        config.rms_norm_eps = vit_config.layer_norm_eps
        config.attention_dropout = vit_config.attention_dropout
        # config.hidden_dropout = vit_config.hidden_dropout
        config.norm_type = config.vit_norm_type
        config.attention_bias = True
        config.mlp_bias = True
        config.use_mla = False
        config.num_experts = 1
        config._attn_implementation = "eager"
        
        yield
        config.hidden_act = hidden_act
        config.hidden_size = hidden_size
        config.intermediate_size = ffn_hidden_size
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads
        config.attention_head_dim = attention_head_dim
        config.use_qk_norm = use_qk_norm
        config.use_rotary_pos_emb = use_rotary_pos_emb
        config.num_hidden_layers = num_hidden_layers
        config.rms_norm_eps = rms_norm_eps
        config.attention_dropout = attention_dropout
        # config.hidden_dropout = hidden_dropout
        config.attention_bias = attention_bias
        config.mlp_bias = mlp_bias
        config.norm_type = norm_type
        config.use_mla = use_mla
        config.num_experts = num_experts
        config._attn_implementation = _attn_implementation

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        
        hidden_states, attention_mask, img_pos = self.embeddings(pixel_values)  
        attention_mask = attention_mask.int()
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length = 0  

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
            )

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        return hidden_states, img_pos


class AnyResVitTransformer(NaVitTransformer):
    def __init__(self, config: HunYuanConfig, vit_config: CLIPVisionConfig, anyres_vit_max_image_size):
        super().__init__(config, vit_config)
        old_anyres_vit_max_image_size = vit_config.max_image_size
        anyres_vit_max_image_size = anyres_vit_max_image_size or old_anyres_vit_max_image_size
        vit_config.max_image_size = anyres_vit_max_image_size
        vit_config.torch_dtype = config.torch_dtype
        vit_config.anyres_vit_two_views = config.anyres_vit_two_views
        vit_config.vit_remove_prenorm = config.vit_remove_prenorm
        vit_config.vit_add_patchemb_bias = config.vit_add_patchemb_bias
        self.embeddings = AnyResCLIPVisionEmbeddings(vit_config)
        vit_config.max_image_size = old_anyres_vit_max_image_size

    def fix_embeddings_fn(self, pixel_values):
        # (B, L, E)
        embeddings, hw = self.embeddings.forward_single(pixel_values)
        embeddings = self.embeddings.pre_layernorm(embeddings)
        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: HunYuanConfig, vit_config: CLIPVisionConfig):
        super().__init__()
        embed_dim = vit_config.hidden_size

        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=vit_config.layer_norm_eps)
        self.embeddings = CLIPVisionEmbeddings(vit_config, skip_cls_token=config.skip_cls_token, vit_patch=config.vit_patch)

        with self.prepare_args(config, vit_config):
            self.layers = nn.ModuleList(
                [HunYuanDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )

    @contextmanager
    def prepare_args(self, config, vit_config):
        hidden_act = config.hidden_act
        hidden_size = config.hidden_size
        ffn_hidden_size = config.intermediate_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        attention_head_dim = config.attention_head_dim
        use_qk_norm = config.use_qk_norm
        use_rotary_pos_emb =  config.use_rotary_pos_emb
        num_hidden_layers = config.num_hidden_layers   
        rms_norm_eps = config.rms_norm_eps
        attention_dropout = config.attention_dropout
        # hidden_dropout = config.hidden_dropout
        norm_type = config.norm_type
        attention_bias = config.attention_bias
        mlp_bias = config.mlp_bias
        use_mla = config.use_mla
        num_experts = config.num_experts
        _attn_implementation = config._attn_implementation

        config.hidden_act = vit_config.hidden_act
        config.hidden_size = vit_config.hidden_size
        config.intermediate_size = vit_config.intermediate_size
        config.num_attention_heads = vit_config.num_attention_heads
        config.num_key_value_heads = None
        config.attention_head_dim = vit_config.hidden_size // vit_config.num_attention_heads
        config.use_qk_norm = False
        config.use_rotary_pos_emb = False
        config.num_hidden_layers = vit_config.num_hidden_layers
        config.rms_norm_eps = vit_config.layer_norm_eps
        config.attention_dropout = vit_config.attention_dropout
        # config.hidden_dropout = 0.0
        config.norm_type = "fused"
        config.attention_bias = True
        config.mlp_bias = True
        config.use_mla = False
        config.num_experts = 1
        config._attn_implementation = "eager"

        yield

        config.hidden_act = hidden_act
        config.hidden_size = hidden_size
        config.intermediate_size = ffn_hidden_size
        config.num_attention_heads = num_attention_heads
        config.num_key_value_heads = num_key_value_heads
        config.attention_head_dim = attention_head_dim
        config.use_qk_norm = use_qk_norm
        config.use_rotary_pos_emb = use_rotary_pos_emb
        config.num_hidden_layers = num_hidden_layers
        config.rms_norm_eps = rms_norm_eps
        config.attention_dropout = attention_dropout
        # config.hidden_dropout = hidden_dropout
        config.norm_type = norm_type
        config.attention_bias = attention_bias
        config.mlp_bias = mlp_bias
        config.use_mla = use_mla
        config.num_experts = num_experts
        config._attn_implementation = _attn_implementation

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        img_index=None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, img_index=img_index)
        hidden_states = self.pre_layrnorm(hidden_states)
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        attention_mask = torch.ones(batch, 1, seq_len, seq_len, dtype=torch.float32, device=device)

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs[0]

        return hidden_states


class Vit(torch.nn.Module):
    def __init__(self, config, resampler_token=64, pool_rate=2):
        super().__init__()
        self.config = config
        self.vit_mapping_type = config.vit_mapping_type
        self.anyres_vit_max_image_size = config.anyres_vit_max_image_size
        self.skip_cls_token = config.skip_cls_token
        self.pool_rate = pool_rate
        self.vit_type = self.config.vit_type
        self.anyres_vit_two_views = self.config.anyres_vit_two_views
        if self.vit_type in ['Vit-g', 'Vit-bigG', 'NaVit', 'EvaVit', 'AnyResVit']:
            self.img_init(resampler_token, config.vit_input_resolution, config.vit_mapping_type, pool_rate)
        else:
            raise NotImplementedError(f"unsupported vit type: {self.vit_type}")
    
    def img_init(self, resampler_token=64, vit_input_resolution=224, vit_mapping_type='resampler', pool_rate=2):
        if self.vit_type == 'AnyResVit':
            vit_config = json.load(open(f"{self.config.vit_path}/config.json"))
            self.vit_config = types.SimpleNamespace(**vit_config["vision_config"])
            self.vit_config.image_size = vit_input_resolution
            self.vit = AnyResVitTransformer(self.config, self.vit_config, self.anyres_vit_max_image_size)
        elif self.vit_type == 'Vit-g':
            vit_config = json.load(open(f"{self.config.vit_path}/config.json"))
            self.vit_config = types.SimpleNamespace(**{**vit_config["vision_config_dict"],**vit_config["vision_config"]})
            self.vit_config.vit_input_resolution = vit_input_resolution
            self.vit = CLIPVisionTransformer(self.config, self.vit_config)  
        else:
           assert False, "other vit_types are not supported"

        if self.vit_mapping_type == 'simple_conv_mlp':
            self.perceive = SimpleConvMlp(self.vit_config.hidden_size, self.config.hidden_size, self.config.anyres_pooling_size, \
                self.config.vit_used_rms_norm, self.config.rms_norm_eps, poolmlp=False, twoview=True)
        elif self.vit_mapping_type == 'oryx_mlp':
            self.perceive = OryxMLPv2(self.vit_config.hidden_size, self.config.hidden_size, twoview=True, use_pe=False)
        elif self.vit_mapping_type == 'mlp':
            self.mlp_depth = 2
            # one mlp layer already in gpt_model.py
            mlp_hidden_size = self.vit_config.hidden_size
            if self.vit_type in ['NaVit', 'EvaVit']:
                mlp_hidden_size *= self.vit_config.adaptor_patch_size **2
            if self.mlp_depth > 1:
                mlp_modules = [torch.nn.Linear(mlp_hidden_size, self.config.hidden_size), torch.nn.GELU()]
                if self.vit_type in ['NaVit', 'EvaVit']:
                    for _ in range(1, self.mlp_depth):
                        mlp_modules.append(torch.nn.Linear(self.config.hidden_size, self.config.hidden_size))
                        mlp_modules.append(torch.nn.GELU())
                self.perceive = torch.nn.Sequential(*mlp_modules)
        else:
           assert False, "other vit_mapping_types are not supported"

        self.vit_patch_mlp = (self.config.vit_patch > 1 and self.vit_mapping_type == 'mlp') or self.config.vit_patch == 0     
        for name, param in self.named_parameters():
            setattr(param, "is_vit_param", True)

    def forward(self, images, img_index=None):
        if self.vit_type in ['AnyResVit']:
            dtype = self.config.torch_dtype
            device = torch.cuda.current_device()

            images_size = []
            for i in range(len(images)):
                images_size.append([])
                for j in range(len(images[i])): 
                    images_size[i].append((images[i][j].size()[1] // self.vit_config.patch_size, images[i][j].size()[2] // self.vit_config.patch_size))
        
            images_feats, img_batch_pos = self.vit(pixel_values=images)
            a2 = self.vit_config.adaptor_patch_size ** 2

            if self.anyres_vit_two_views:
                step = 2
            else:
                step = 1
            perceive_fn = lambda x, img_size, is_video: self.perceive(x, img_size, is_video=is_video)     
            images_list = []
            images_fix_i = 0
            num_img_batch_pos = len(img_batch_pos)
            for i in range(num_img_batch_pos): # batch_id
                for j in range(0, len(img_batch_pos[i]), step): 
                    if self.anyres_vit_two_views:
                        lower_idx, lower_begin, lower_end = img_batch_pos[i][j]
                        lower_begin = lower_begin * a2
                        lower_end = lower_end * a2
                        higher_idx, higher_begin, higher_end = img_batch_pos[i][j + 1]
                        higher_begin = higher_begin * a2
                        higher_end = higher_end * a2
                        lower_res_feat = images_feats[lower_idx, lower_begin:lower_end].unsqueeze(0)
                        higher_res_feat = images_feats[higher_idx, higher_begin:higher_end].unsqueeze(0)
                        lower_images_size = images_size[i][j]
                        higher_images_size = images_size[i][j + 1]
                        images_list.append(self.perceive(lower_res_feat, lower_images_size, higher_res_feat, higher_images_size))
                    else:
                        idx, begin, end = img_batch_pos[i][j]
                        begin = begin * a2
                        end = end * a2
                        is_video = hasattr(images[i][j],'_is_video') and images[i][j]._is_video 
                        images_list.append(perceive_fn(images_feats[idx, begin:end].unsqueeze(0), images_size[i][j], is_video=is_video))

            images = torch.cat(images_list, dim=1)

            new_batch_pos = []
            k = 0; cur_len = 0
            for i in range(len(images_size)):
                new_batch_pos.append([])
                for j in range(0, len(images_size[i]), step):
                    new_pos = [0, cur_len, cur_len + images_list[k].size(1)]
                    cur_len += images_list[k].size(1)
                    k += 1
                    new_batch_pos[i].append(new_pos)
            return images, new_batch_pos
        elif self.vit_type == 'Vit-g':
            images = self.vit(pixel_values=images, interpolate_pos_encoding=False, img_index=img_index)
        else: 
            assert False, "other vit_types are not supported"
        
        if self.vit_mapping_type == 'mlp':
            if self.vit_type in ['Vit-g'] and not self.skip_cls_token:
                images = images[:,1:,:]
            b, v, d = images.shape
            s = int(math.sqrt(v))
            images = images.reshape(b, s, s, d)


            if self.vit_patch_mlp and img_index is not None:
                L_tensor = torch.tensor(img_index)
                device = images.device
                # 获取子图位置
                nonzero_indices = torch.nonzero(L_tensor).squeeze().to(device)
                # 获取主图位置
                zero_indices = torch.nonzero(L_tensor == 0).squeeze().to(device)


                images_nonzero = torch.index_select(images,0, nonzero_indices).to(device)
                images_zero = torch.index_select(images, 0, zero_indices).to(device)

                # 子图额外多pool一次
                pool_rate = self.pool_rate * 2
                images_nonzero = images_nonzero.reshape(-1, s // pool_rate, pool_rate, s // pool_rate, pool_rate, d)
                images_nonzero = images_nonzero.permute(0, 1, 3, 5, 2, 4).reshape(-1, (s // pool_rate) * (s // pool_rate), d,
                                                                      pool_rate*pool_rate).mean(-1)

                # 为了组batch折衷方案
                images_nonzero = F.pad(images_nonzero, (0, 0, 0, (s // self.pool_rate) * (s // self.pool_rate)- (s // pool_rate) * (s // pool_rate)))
                images_zero = images_zero.reshape(-1, s // self.pool_rate, self.pool_rate, s // self.pool_rate, self.pool_rate, d)
                images_zero = images_zero.permute(0, 1, 3, 5, 2, 4).reshape(-1, (s // self.pool_rate) * (s // self.pool_rate), d,
                                                                  self.pool_rate*self.pool_rate).mean(-1)
                # 组batch
                images = torch.zeros(b, (s // self.pool_rate) * (s // self.pool_rate), d).to(device).to(images.dtype)
                images.index_copy_(0, nonzero_indices, images_nonzero)
                images.index_copy_(0, zero_indices, images_zero)

                if self.mlp_depth >= 2:
                    images = self.perceive(images)
            else:
                if s % self.pool_rate == 0:
                    images = images.reshape(b, s//self.pool_rate, self.pool_rate, s//self.pool_rate, self.pool_rate, d)
                    images = images.permute(0, 1, 3, 5, 2, 4).reshape(b, (s//self.pool_rate) * (s//self.pool_rate), d, -1).mean(-1)
                    if self.mlp_depth >= 2:
                        images = self.perceive(images)
                else:
                    raise ValueError
        return images


class SimpleConvMlp(nn.Module):
    def __init__(self, in_channels, out_channels, anyres_pooling_size, vit_used_rms_norm, rms_norm_eps, twoview=False, poolmlp=True, cat_extra_token=True):
        super().__init__()

        embed_std = 1 / math.sqrt(out_channels)
        if poolmlp:
            # if args.learnable_mlp_pooling_size is not None:
            #     in_channels *= args.learnable_mlp_pooling_size ** 2
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.GELU()
            )
            self.vit_linear_encoder = nn.Linear(out_channels, out_channels)
            self.image_newline = nn.Parameter(
                torch.randn(out_channels) * embed_std
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=anyres_pooling_size, stride=anyres_pooling_size),
                nn.GELU(),
                nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=1),
            )
            self.mlp = nn.Linear(in_channels * 4, out_channels)
            self.image_newline = nn.Parameter(
                torch.randn(in_channels * 4) * embed_std
            )
        self.poolmlp = poolmlp

        self.image_begin = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        self.image_end = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        
        if twoview:
            self.image_sep = nn.Parameter(
                torch.randn(out_channels) * embed_std
            )

        self.cat_extra_token = cat_extra_token
        self.use_rms_norm = vit_used_rms_norm
        if self.use_rms_norm:
            self.before_rms = HunYuanRMSNorm(in_channels, eps=rms_norm_eps)
            self.after_rms = HunYuanRMSNorm(out_channels, eps=rms_norm_eps)

    def forward(self, x, size=(16,16), x2=None, size2=(16, 16), is_video=False):
        return self.single_forward(x=x, size=size, x2=x2, size2=size2, is_video=is_video)

    def single_forward(self, x, size=(16,16), x2=None, size2=(16, 16), is_video=False): 
        remove_vit_special_tokens = False
        learnable_mlp_pooling_size = None         
        if self.use_rms_norm:
            x = self.before_rms(x)            
        h, w = size
        dtype = x.dtype
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)
        if self.poolmlp:
            if learnable_mlp_pooling_size is None:
                x = F.avg_pool2d(x, anyres_pooling_size)
                x = self.proj(x.permute(0, 2, 3, 1)) # b, h, w, c
            else:
                x = x.permute(0, 2, 3, 1)   # b, h, w, c
                x = x.reshape(x.shape[0], h // learnable_mlp_pooling_size, learnable_mlp_pooling_size, 
                                    w // learnable_mlp_pooling_size, learnable_mlp_pooling_size, -1)
                x = x.permute(0, 1, 3, 2, 4, 5).reshape(x.shape[0], h // learnable_mlp_pooling_size, w // learnable_mlp_pooling_size, -1)
                x = self.proj(x)
            x = self.vit_linear_encoder(x)
            b, h, w, c = x.shape
            if not remove_vit_special_tokens:
                x = torch.cat([
                    x,
                    self.image_newline.reshape(1, 1, 1, c).expand(b, h, 1, c).to(dtype, non_blocking=True)
                ], dim=2)
            x = x.reshape(b, -1, c)
        else:
            x = self.proj(x) #b,c,h,w
            if is_video:
                video_avgpool_size = 2
                stride = 2
                x = F.avg_pool2d(x, kernel_size = video_avgpool_size, stride = stride)
            b, c, h, w = x.shape
            if not remove_vit_special_tokens:
                x = torch.cat([
                    x,
                    self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype, non_blocking=True)
                ], dim=-1)
            x = x.reshape(b, c, -1).permute(0, 2, 1)
            x = self.mlp(x)


        if x2 is not None:
            h2, w2 = size2
            x2 = x2.permute(0, 2, 1).reshape(x2.shape[0], -1, h2, w2)
            if self.poolmlp:
                x2 = F.avg_pool2d(x2, 2)
                x2 = self.proj(x2.permute(0, 2, 3, 1)) # b, h, w, c
                x2 = self.vit_linear_encoder(x2)
                b2, h2, w2, c2 = x2.shape
                if not remove_vit_special_tokens:
                    x2 = torch.cat([
                        x2,
                        self.image_newline.reshape(1, 1, 1, c2).expand(b2, h2, 1, c2).to(dtype, non_blocking=True)
                    ], dim=2)
                x2 = x2.reshape(b2, -1, c2)
            else:
                x2 = self.proj(x2)
                b2, c2, h2, w2 = x2.shape
                if not remove_vit_special_tokens:
                    x2 = torch.cat([
                        x2,
                        self.image_newline.reshape(1, c2, 1, 1).expand(b2, c2, h2, 1).to(dtype, non_blocking=True)
                    ], dim=-1)
                x2 = x2.reshape(b2, c2, -1).permute(0, 2, 1) #b,n,c
                x2 = self.mlp(x2)

            sep = self.image_sep.reshape(1, 1, -1).expand(b2, 1, x2.shape[-1]).to(dtype, non_blocking=True)

            x = torch.cat([x, sep, x2], dim=1)
        
        if self.cat_extra_token:
            begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
            end = self.image_end.reshape(1, 1, -1).expand(b, 1, x.shape[-1]).to(dtype, non_blocking=True)
            x = torch.cat([begin, x, end], dim=1)

        if self.use_rms_norm:
            return self.after_rms(x)
        else:
            return x


class NormalizedDwPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.predictor = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x, forward_type='2x'):
        B, H, W, C = x.shape

        if forward_type == '2x':
            new_x = x.reshape(B, H//2, 2, W//2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H//2, W//2, 4, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
            fused_x = torch.cat([new_x, pooled_x], dim=-1)
        elif forward_type == '1x':
            new_x = x.reshape(B, H, W, 1, C)
            fused_x = torch.cat([new_x, new_x], dim=-1)
        elif forward_type == '4x':
            new_x = x.reshape(B, H//4, 4, W//4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H//4, W//4, 16, C)
            pooled_x = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 16, -1)
            fused_x = torch.cat([new_x, pooled_x], dim=-1)
        
        score = self.predictor(fused_x)
        normalized_score = F.softmax(score, dim=-2)
        new_x = (new_x * normalized_score).sum(dim=-2)
        return new_x


class OryxMLPv2(nn.Module):
    def __init__(self, in_channels, out_channels, twoview=False, use_pe=False):
        super().__init__()
        
        self.proj1 = nn.Linear(in_channels, out_channels)
        self.proj2 = nn.Linear(out_channels, out_channels)
        self.act = nn.GELU()
        self.pooler = NormalizedDwPooler(out_channels)
        embed_std = 1 / math.sqrt(out_channels)

        self.use_pe = use_pe
        if not use_pe:
            self.image_newline = nn.Parameter(
                torch.randn(out_channels) * embed_std
            )
        self.image_begin = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        self.image_end = nn.Parameter(
            torch.randn(out_channels) * embed_std
        )
        
        if twoview:
            self.image_sep = nn.Parameter(
                torch.randn(out_channels) * embed_std
            )

    def forward(self, x, size=(16,16), x2=None, size2=(16, 16), is_video=False):
        h, w = size
        dtype = x.dtype
        x = x.reshape(x.shape[0], h, w, -1)
        # x = self.pooler(x, forward_type=REGIONAL_POOL)
        # x = self.proj(x) #b,h,w, c
        x = self.proj1(x)
        x = self.pooler(x, forward_type='2x')
        x = self.act(x)
        x = self.proj2(x)


        b, h, w, c = x.shape
        if not self.use_pe:
            x = torch.cat([
                x,
                self.image_newline.reshape(1, 1, 1, c).expand(b, h, 1, c).to(dtype)
            ], dim=2)
        else:
            pe_h = torch.arange(h, dtype=torch.long, device=x.device).reshape(1, h, 1, 1).expand(b, h, w, 1).reshape(b, h*w, 1)
            pe_w = torch.arange(w, dtype=torch.long, device=x.device).reshape(1, 1, w, 1).expand(b, h, w, 1).reshape(b, h*w, 1)
            pe = torch.cat([pe_h, pe_w], dim=-1)

        x = x.reshape(b, -1, c)

        if x2 is not None:
            h2, w2 = size2
            x2 = x2.reshape(x2.shape[0], h2, w2, -1)
            # x2 = self.pooler(x2, forward_type=REGIONAL_POOL)
            ## x2 = self.proj(x2) #b,h,w, c
            x2 = self.proj1(x2)
            x2 = self.pooler(x2, forward_type='2x')
            x2 = self.act(x2)
            x2 = self.proj2(x2)

            b2, h2, w2, c2 = x2.shape
            if not self.use_pe:
                x2 = torch.cat([
                    x2,
                    self.image_newline.reshape(1, 1, 1, c).expand(b, h2, 1, c).to(dtype)
                ], dim=2)
            x2 = x2.reshape(b, -1, c)
            sep = self.image_sep.reshape(1, 1, -1).expand(b, 1, c2).to(dtype)
            x = torch.cat([x, sep, x2], dim=1)
        
        begin = self.image_begin.reshape(1, 1, -1).expand(b, 1, c).to(dtype)
        end = self.image_end.reshape(1, 1, -1).expand(b, 1, c).to(dtype)
        x = torch.cat([begin, x, end], dim=1)
        # print(x.shape, x2.shape, h, w, h2, w2)
        # print("vit rank = " + str(torch.distributed.get_rank()) +" x = " + str(x))
        if self.use_pe:
            zero_pad = torch.zeros(b, 1, 2, device=x.device, dtype=torch.long)
            pe = torch.cat([zero_pad, pe, zero_pad], dim=1)
            assert pe.shape[1] == x.shape[1]
            return x, pe
        else:
            nseq = x.shape[1]
            fake_pe = torch.zeros(b, nseq, 2, device=x.device, dtype=torch.long)
            return x #, fake_pe

