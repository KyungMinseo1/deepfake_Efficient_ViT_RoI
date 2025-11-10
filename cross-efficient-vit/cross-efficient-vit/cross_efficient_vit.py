
import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from efficient_net.efficientnet_pytorch import EfficientNet
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
import os, re

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
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value # type: ignore

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
        for attn, ff in self.layers: # type: ignore
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
    def __init__(self, sm_dim, roi_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # sm ↔ roi
                ProjectInOut(sm_dim, roi_dim, PreNorm(roi_dim, Attention(roi_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(roi_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                
                # roi ↔ lg
                ProjectInOut(roi_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, roi_dim, PreNorm(roi_dim, Attention(roi_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                
                # sm ↔ lg
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
            ]))

    def forward(self, sm_tokens, roi_tokens, lg_tokens):
        # 각 토큰을 CLS와 patch로 분리
        (sm_cls, sm_patch_tokens) = (sm_tokens[:, :1], sm_tokens[:, 1:])
        (roi_cls, roi_patch_tokens) = (roi_tokens[:, :1], roi_tokens[:, 1:])
        (lg_cls, lg_patch_tokens) = (lg_tokens[:, :1], lg_tokens[:, 1:])

        for sm_attend_roi, roi_attend_sm, roi_attend_lg, lg_attend_roi, sm_attend_lg, lg_attend_sm in self.layers: # type: ignore
            # sm & roi
            sm_cls_new = sm_attend_roi(sm_cls, context=roi_patch_tokens, kv_include_self=True)
            roi_cls_new = roi_attend_sm(roi_cls, context=sm_patch_tokens, kv_include_self=True)
            
            # roi & lg
            roi_cls_new = roi_cls_new + roi_attend_lg(roi_cls, context=lg_patch_tokens, kv_include_self=True)
            lg_cls_new = lg_attend_roi(lg_cls, context=roi_patch_tokens, kv_include_self=True)
            
            # sm & lg
            sm_cls_new = sm_cls_new + sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True)
            lg_cls_new = lg_cls_new + lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True)
            
            # Residual connection
            sm_cls = sm_cls + sm_cls_new
            roi_cls = roi_cls + roi_cls_new
            lg_cls = lg_cls + lg_cls_new

        # CLS와 patch 토큰 다시 합치기
        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        roi_tokens = torch.cat((roi_cls, roi_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        
        return sm_tokens, roi_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        roi_dim,
        lg_dim,
        sm_enc_params,
        roi_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params), # type: ignore
                Transformer(dim = roi_dim, dropout = dropout, **roi_enc_params), # type: ignore
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params), # type: ignore
                CrossTransformer(sm_dim = sm_dim, roi_dim = roi_dim, lg_dim=lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, roi_tokens, lg_tokens):
        for sm_enc, roi_enc, lg_enc, cross_attend in self.layers: # type: ignore
            sm_tokens, roi_tokens, lg_tokens = sm_enc(sm_tokens), roi_enc(roi_tokens), lg_enc(lg_tokens)
            sm_tokens, roi_tokens, lg_tokens = cross_attend(sm_tokens, roi_tokens, lg_tokens)

        return sm_tokens, roi_tokens, lg_tokens

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
        is_roi = False,
        efficient_net = 0
    ):
        super().__init__()
        # assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        if efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
            checkpoint = torch.load("pretrained_model/efficientnet-b7.pth", map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)

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
      
        max_patches = max((image_size // patch_size) ** 2, 98)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        # Region of Interest
        self.is_roi = is_roi
    
    # extract ROI
    def extract_patches_roi(self, imgs, scaled_landmarks):
        """
        imgs: [B, C', H', W'] (CNN feature map)
        scaled_landmarks: [B, N_coords, 2] (Tensor, feature map scale)
        """
        B, N_coords, _ = scaled_landmarks.shape
        patch_size = self.patch_size # ROI Align 출력 크기 (예: 7)

        # Feature map 크기
        H_feat, W_feat = imgs.shape[2:]

        # 랜드마크에서 ROI 좌표 계산 (Tensor 연산)
        center = scaled_landmarks
        half_patch = patch_size / 2
        
        x1 = center[:, :, 0] - half_patch
        y1 = center[:, :, 1] - half_patch
        x2 = center[:, :, 0] + half_patch
        y2 = center[:, :, 1] + half_patch

        x1 = torch.clamp(x1, min=0, max=W_feat - patch_size) # x1은 x2보다 작아야 함 (최대 W_feat - patch_size)
        y1 = torch.clamp(y1, min=0, max=H_feat - patch_size) # y1은 y2보다 작아야 함 (최대 H_feat - patch_size)
        
        x2 = torch.clamp(x2, min=patch_size, max=W_feat)
        y2 = torch.clamp(y2, min=patch_size, max=H_feat)
        
        # 배치 인덱스 생성
        batch_indices = torch.arange(B, device=imgs.device).repeat_interleave(N_coords)
        batch_indices = batch_indices.unsqueeze(1).float()
        
        # [B * N_coords, 4] 형태의 좌표 Tensor 생성
        coords_flat = torch.stack((x1.flatten(), y1.flatten(), x2.flatten(), y2.flatten()), dim=1)
        
        # 최종 rois Tensor: [B*N_coords, 5]
        rois = torch.cat((batch_indices, coords_flat), dim=1)

        # Extract patches using ROI Align
        patches = roi_align(imgs, rois, output_size=(patch_size, patch_size)) # type: ignore
        return patches

    def forward(self, img, landmarks=None, is_sample=False):
        """
        img: [B, C, H, W]
        landmarks: list of [(x, y), ...] for each image (len = 64)
        """
        # CNN feature extraction
        x = self.efficient_net.extract_features_at_block(img, self.efficient_block)  
        # 여기서 x: [B, C', H', W']
        print(x.shape)

        if self.is_roi and landmarks is not None:
            # landmarks는 [B, N_coords, 2] 형태의 Tensor라고 가정

            B, C, H_feat, W_feat = x.shape
            _, _, H_img, W_img = img.shape
            
            # 스케일 보정 (Tensor 연산)
            scale_x = W_feat / W_img
            scale_y = H_feat / H_img
            
            # [1, 1, 2] 형태의 스케일 Tensor 생성
            scale_tensor = torch.tensor([scale_x, scale_y], device=landmarks.device).view(1, 1, 2)
            
            # Tensor 곱셈으로 전체 배치의 랜드마크 스케일 보정
            scaled_landmarks = landmarks * scale_tensor
            
            # ROI-based patch extraction
            patches = self.extract_patches_roi(x, scaled_landmarks)

            if is_sample:
                os.makedirs('sample', exist_ok=True)
                with torch.no_grad():
                    # -----------------------------
                    # 1️⃣ Feature map + landmarks
                    # -----------------------------
                    feat_map = x[0].detach().cpu()  # 첫 번째 이미지
                    feat_map_vis = feat_map.mean(0).numpy()  # [H', W']
                    XY = scaled_landmarks[0].detach().cpu().numpy()  # 첫 번째 이미지의 좌표

                    plt.figure(figsize=(6,6))
                    plt.imshow(feat_map_vis, cmap='viridis', origin='upper')
                    plt.scatter(XY[:,0], XY[:,1], s=10, c='red')
                    plt.title('Feature map with scaled landmarks')
                    plt.savefig("sample/featuremap_with_landmarks.png")
                    plt.close()

                    # -----------------------------
                    # 2️⃣ Extracted ROI patches 시각화
                    # -----------------------------

                    # 첫 번째 배치의 패치만
                    B, N_coords, C, p, _ = 1, XY.shape[0], patches.shape[1], patches.shape[2], patches.shape[3]
                    patches_for_sample = patches[:N_coords].detach().cpu()

                    # 채널 평균
                    patch_imgs = patches_for_sample.mean(1).numpy()  # [N_coords, 7, 7]

                    # -----------------------------
                    # 3️⃣ Grid로 보기 (ex: 10x10)
                    # -----------------------------
                    num_show = min(N_coords, 100)  # 너무 많으면 100개만
                    grid_size = int(num_show ** 0.5)

                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8,8))
                    for i, ax in enumerate(axes.flat):
                        ax.imshow(patch_imgs[i], cmap='viridis')
                        ax.axis('off')
                    plt.suptitle('Extracted ROI patches (mean over channels)')
                    plt.tight_layout()
                    plt.savefig("sample/roi_patches_grid.png")
                    plt.close()

            # Flatten each patch
            patches = patches.flatten(1)  # [B*num_coords, C'*7*7]

            # Linear embedding
            x = self.linear_proj(patches)

            # Reshape to [B, N, dim]
            num_coords = len(landmarks[0])  # 121
            B = img.size(0)
            x = x.view(B, num_coords, -1)

            # Add [CLS] token
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :x.size(1)]

        else:
            if is_sample:
                feat_map = x[0].detach().cpu() # [C', H', W']
                num_show = min(feat_map.shape[0], 100)  # 너무 많으면 100개만
                grid_size = int(num_show ** 0.5)
                fig, axes = plt.subplots(grid_size, grid_size, figsize=(8,8))
                for i, ax in enumerate(axes.flat):
                    if i < num_show:
                        ax.imshow(feat_map[i], cmap='viridis')
                    else:
                        ax.axis('off')  # 남는 칸은 비우기
                    ax.axis('off')
                plt.suptitle('Extracted Small patches')
                plt.tight_layout()
                plt.savefig("sample/sm_patches_grid.png", dpi=200)
                plt.close()

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
        config,
        is_sample = False,
        efficient_net = 0
    ):
        super().__init__()
        image_size = config['model']['image-size']
        num_classes = config['model']['num-classes'] 
        sm_dim = config['model']['sm-dim']
        sm_channels = config['model']['sm-channels']
        roi_dim = config['model']['roi-dim']
        roi_channels = config['model']['roi-channels']         
        lg_dim = config['model']['lg-dim']
        lg_channels = config['model']['lg-channels']         
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
        lg_patch_size = config['model']['lg-patch-size']
        lg_enc_depth = config['model']['lg-enc-depth'] 
        lg_enc_mlp_dim = config['model']['lg-enc-mlp-dim']
        lg_enc_heads = config['model']['lg-enc-heads']
        lg_enc_dim_head = config['model']['lg-enc-dim-head']
        cross_attn_depth = config['model']['cross-attn-depth']
        cross_attn_heads = config['model']['cross-attn-heads']
        cross_attn_dim_head = config['model']['cross-attn-dim-head']
        depth = config['model']['depth']
        dropout = config['model']['dropout']
        emb_dropout = config['model']['emb-dropout']

        self.is_sample = is_sample
        self.e_net = efficient_net

        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout, efficient_block = 72, channels=sm_channels, is_roi=False, efficient_net=self.e_net)
        self.roi_image_embedder = ImageEmbedder(dim = roi_dim, image_size = image_size, patch_size = roi_patch_size, dropout = emb_dropout, efficient_block = 1, channels=roi_channels, is_roi=True, efficient_net=self.e_net)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout, efficient_block = 1, channels=lg_channels, is_roi=False, efficient_net=self.e_net)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            roi_dim = roi_dim,
            lg_dim = lg_dim,
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
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.roi_mlp_head = nn.Sequential(nn.LayerNorm(roi_dim), nn.Linear(roi_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img, coordinates):
        sm_tokens = self.sm_image_embedder(img, is_sample = self.is_sample)
        roi_tokens = self.roi_image_embedder(img, coordinates, is_sample = self.is_sample)
        lg_tokens = self.lg_image_embedder(img, is_sample = self.is_sample)

        sm_tokens, roi_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, roi_tokens, lg_tokens)

        sm_cls, roi_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, roi_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        roi_logits = self.roi_mlp_head(roi_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + roi_logits + lg_logits