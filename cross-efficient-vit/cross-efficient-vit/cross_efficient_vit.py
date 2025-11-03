
import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from efficient_net.efficientnet_pytorch import EfficientNet
from torchvision.ops import roi_align

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, roi_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, roi_dim, PreNorm(roi_dim, Attention(roi_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(roi_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, roi_tokens):
        (sm_cls, sm_patch_tokens), (roi_cls, roi_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, roi_tokens))

        for sm_attend_roi, roi_attend_sm in self.layers:
            sm_cls = sm_attend_roi(sm_cls, context = roi_patch_tokens, kv_include_self = True) + sm_cls
            roi_cls = roi_attend_sm(roi_cls, context = sm_patch_tokens, kv_include_self = True) + roi_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        roi_tokens = torch.cat((roi_cls, roi_patch_tokens), dim = 1)
        return sm_tokens, roi_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        roi_dim,
        sm_enc_params,
        roi_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = roi_dim, dropout = dropout, **roi_enc_params),
                CrossTransformer(sm_dim = sm_dim, roi_dim = roi_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, roi_tokens):
        for sm_enc, roi_enc, cross_attend in self.layers:
            sm_tokens, roi_tokens = sm_enc(sm_tokens), roi_enc(roi_tokens)
            sm_tokens, roi_tokens = cross_attend(sm_tokens, roi_tokens)

        return sm_tokens, roi_tokens

# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.,
        efficient_block = 8,
        channels,
        is_roi = False
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficient_net.delete_blocks(efficient_block)
        self.efficient_block = efficient_block
        
        for index, (name, param) in enumerate(self.efficient_net.named_parameters()):
            param.requires_grad = True
                
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.linear_proj = nn.Linear(patch_dim, dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            self.linear_proj
        )
      
        max_patches = max((image_size // patch_size) ** 2, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        # Region of Interest
        self.is_roi = is_roi
    
    # extract ROI
    def extract_patches_roi(self, imgs, coords_batch):
        """
        imgs: [B, C, H, W]
        coords_batch: list of [(x, y), ...] for each image
        """
        rois = []
        for b, coords in enumerate(coords_batch):
            for (x, y) in coords:
                x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
                x2, y2 = x + self.patch_size // 2, y + self.patch_size // 2
                rois.append([b, x1, y1, x2, y2])
        rois = torch.tensor(rois, dtype=torch.float32, device=imgs.device)

        # Extract 7x7 patches using ROI Align
        patches = roi_align(imgs, rois, output_size=(self.patch_size, self.patch_size))
        return patches  # [B * num_coords, C, 7, 7]

    def forward(self, img, landmarks=None):
        """
        img: [B, C, H, W]
        landmarks: list of [(x, y), ...] for each image (len = 64)
        """
        # CNN feature extraction
        x = self.efficient_net.extract_features_at_block(img, self.efficient_block)  
        # 여기서 x: [B, C', H', W']

        if self.is_roi and landmarks is not None:
            # 스케일 보정
            B, C, H_feat, W_feat = x.shape
            _, _, H_img, W_img = img.shape
            scale_x = W_feat / W_img
            scale_y = H_feat / H_img
            scaled_landmarks = [
                [(lx * scale_x, ly * scale_y) for (lx, ly) in coords]
                for coords in landmarks
            ]

            # ROI-based patch extraction (using dlib landmarks)
            patches = self.extract_patches_roi(x, scaled_landmarks)  # [B*num_coords, C', 7, 7]

            # Flatten each patch
            patches = patches.flatten(1)  # [B*num_coords, C'*7*7]

            # Linear embedding
            x = self.linear_proj(patches)

            # Reshape to [B, N, dim]
            num_coords = len(landmarks[0])  # usually 64
            B = img.size(0)
            x = x.view(B, num_coords, -1)

            # Add [CLS] token
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :x.size(1)]

        else:
            x = self.to_patch_embedding(x)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        
        return self.dropout(x)

# cross ViT class

class CrossEfficientViT(nn.Module):
    def __init__(
        self,
        *,
        config
    ):
        super().__init__()
        image_size = config['model']['image-size']
        num_classes = config['model']['num-classes'] 
        sm_dim = config['model']['sm-dim']
        sm_channels = config['model']['sm-channels']
        roi_dim = config['model']['roi-dim']
        roi_channels = config['model']['roi-channels']         
        sm_patch_size = config['model']['sm-patch-size']
        sm_enc_depth = config['model']['sm-enc-depth'] 
        sm_enc_heads = config['model']['sm-enc-heads']
        sm_enc_mlp_dim = config['model']['sm-enc-mlp-dim']
        sm_enc_dim_head = config['model']['sm-enc-dim-head']
        roi_patch_size = config['model']['roi-patch-size']
        roi_enc_depth = config['model']['roi-enc-depth'] 
        roi_enc_mlp_dim = config['model']['roi-enc-mlp-dim']
        roi_enc_heads = config['model']['roi-enc-heads']
        roi_enc_dim_head = config['model']['roi-enc-dim-head']
        cross_attn_depth = config['model']['cross-attn-depth']
        cross_attn_heads = config['model']['cross-attn-heads']
        cross_attn_dim_head = config['model']['cross-attn-dim-head']
        depth = config['model']['depth']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']



        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout, efficient_block = 16, channels=sm_channels, is_roi=False)
        self.roi_image_embedder = ImageEmbedder(dim = roi_dim, image_size = image_size, patch_size = roi_patch_size, dropout = emb_dropout, efficient_block = 1, channels=roi_channels, is_roi=True)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            roi_dim = roi_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            roi_enc_params = dict(
                depth = roi_enc_depth,
                heads = roi_enc_heads,
                mlp_dim = roi_enc_mlp_dim,
                dim_head = roi_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.roi_mlp_head = nn.Sequential(nn.LayerNorm(roi_dim), nn.Linear(roi_dim, num_classes))

    def forward(self, img, coordinates):
        sm_tokens = self.sm_image_embedder(img)
        roi_tokens = self.roi_image_embedder(img, coordinates)

        sm_tokens, roi_tokens = self.multi_scale_encoder(sm_tokens, roi_tokens)

        sm_cls, roi_cls = map(lambda t: t[:, 0], (sm_tokens, roi_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        roi_logits = self.roi_mlp_head(roi_cls)

        return sm_logits + roi_logits