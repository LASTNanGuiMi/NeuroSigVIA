"""Deterministic alternative images, one fixed channel montage per outer window.

Adaptive heatmap uses TimesNet-inspired top-1 FFT period folding, not TimesNet's
learned convolutions or top-k aggregation. GAF means GASF, cos(phi_i + phi_j).
All decisions are local to the valid prefix; no labels or dataset statistics.
"""
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

REPRESENTATIONS = ('ordinary_line', 'adaptive_heatmap', 'gaf', 'tivit_grayscale')

def rendering_policy(mode):
    if mode not in REPRESENTATIONS:
        raise ValueError(mode)
    return dict(version='imaging36_v1', representation=mode, image_size=224,
                channels='fixed_order_square_grid_one_panel_per_channel',
                valid_prefix_only=True, axes_colorbar_text=False,
                normalization='per_channel_per_outer_window_minmax_constant_to_midpoint',
                line='ordinary_blue_polyline_white_background',
                adaptive_heatmap='top1_mean_channel_rfft_amplitude_exclude_DC_period=floor(L/f); per_channel_cycles_by_period; padded_cells_gray',
                constant_period='valid_length', gaf='GASF_cos_angle_sum_scaled_minus1_plus1_no_PAA',
                gaf_constant='scaled_zero_GASF_minus_one',
                heatmap_colormap='viridis', gaf_color_limits=[-1,1], resize='nearest',
                period_selection='per_sample_outer_window_no_batch_dependence',
                tivit_grayscale=('TiViT upstream 5faafcd ts2image_transformation per channel: '
                                 'robust scaling (median, IQR+1e-5) over the valid prefix; '
                                 'patch_size=int(sqrt(T)), stride=0.1 of patch (min 1), replicate left pad, unfold; '
                                 'global min-max + 1e-5 then power 0.8; grayscale; no colormap'))

def valid_window(window, valid_length=None):
    x = torch.as_tensor(window).detach().cpu().double().numpy()
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('Expected nonempty [channels,time]')
    n = x.shape[1] if valid_length is None else int(valid_length)
    if not 1 <= n <= x.shape[1]:
        raise ValueError('Invalid valid_length')
    x = x[:, :n].copy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite valid samples')
    return x

def scale_channels(x):
    low=x.min(axis=1,keepdims=True); span=x.max(axis=1,keepdims=True)-low
    return np.divide(x-low,span,out=np.full_like(x,.5),where=span>0)

def dominant_period(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape[-1] < 2:
        return 1
    amplitude=np.abs(np.fft.rfft(x-x.mean(axis=-1,keepdims=True),axis=-1)).mean(axis=0)
    amplitude[0]=0
    if not np.any(amplitude[1:]>0):
        return x.shape[-1]
    index=int(np.argmax(amplitude[1:]))+1
    return max(1,x.shape[-1]//index)

def folded_channels(x):
    period=dominant_period(x)
    n=x.shape[-1]; tail=(-n)%period
    padded=np.pad(x,((0,0),(0,tail)))
    mask=np.pad(np.ones_like(x,dtype=bool),((0,0),(0,tail)),constant_values=False)
    return padded.reshape(x.shape[0],-1,period),mask.reshape(x.shape[0],-1,period),period

def gasf_channels(x):
    z=np.clip(2*scale_channels(x)-1,-1,1)
    sine=np.sqrt(np.maximum(0,1-z*z))
    return np.clip(z[:,:,None]*z[:,None,:]-sine[:,:,None]*sine[:,None,:],-1,1)

def tivit_gray_channels(x, stride=0.1):
    """TiViT (upstream 5faafcd, src/tivit.py ts2image_transformation) before its final resize.

    x: [channels, T] valid prefix. Returns [channels, windows, patch] in [0, 1].
    Each channel is its own univariate series, exactly as TiViT embeds channels separately.
    """
    import torch.nn.functional as F
    t = torch.as_tensor(np.asarray(x, dtype=np.float32).T[None])      # B x T x D, B = 1
    median = t.median(1, keepdim=True)[0]
    q = torch.tensor([0.75, 0.25], dtype=t.dtype)
    q75, q25 = torch.quantile(t, q, dim=1, keepdim=True)
    t = (t - median) / (q75 - q25 + 1e-5)
    t = t.permute(0, 2, 1)                                             # b d t
    T = t.shape[-1]
    patch_size = int(math.sqrt(T))
    stride_len = int(patch_size * stride) or 1
    remainder = (T - patch_size) % stride_len
    pad_left = stride_len - remainder if remainder else 0
    x_pad = F.pad(t, (pad_left, 0), mode="replicate")
    x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)  # b d windows patch
    lo = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    hi = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_2d = torch.pow((x_2d - lo) / (hi - lo + 1e-5), 0.8)
    return x_2d[0].double().numpy()

def render_image(window, *, valid_length=None, image_representation, image_size=224):
    from matplotlib import colormaps
    if image_representation not in REPRESENTATIONS:
        raise ValueError(image_representation)
    x=valid_window(window,valid_length)
    channels,n=x.shape
    columns=math.ceil(math.sqrt(channels)); rows=math.ceil(channels/columns)
    if not isinstance(image_size,int) or image_size < max(rows,columns)*3:
        raise ValueError('Image too small for channel grid')
    canvas=Image.new('RGB',(image_size,image_size),'white')
    unit=scale_channels(x)
    if image_representation=='adaptive_heatmap':
        # Determine period on input amplitudes, render channel-normalized values.
        raw,mask,period=folded_channels(x)
        tail=raw.shape[1]*period-n
        matrices=np.pad(unit,((0,0),(0,tail))).reshape(raw.shape)
    elif image_representation=='gaf':
        matrices=(gasf_channels(x)+1)/2
        mask=np.ones_like(matrices,dtype=bool)
    elif image_representation=='tivit_grayscale':
        matrices=tivit_gray_channels(x)
        mask=np.ones_like(matrices,dtype=bool)
    for ch in range(channels):
        row,col=divmod(ch,columns)
        left=col*image_size//columns; right=(col+1)*image_size//columns-1
        top=row*image_size//rows; bottom=(row+1)*image_size//rows-1
        width,height=right-left,bottom-top
        if image_representation=='ordinary_line':
            panel=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(panel)
            points=[(float(t),float(y)) for t,y in zip(np.linspace(0,width-1,n),(1-unit[ch])*(height-3)+1)]
            if n==1:
                draw.line([(0,points[0][1]),(width-1,points[0][1])],fill=(31,119,180),width=1)
            else:
                draw.line(points,fill=(31,119,180),width=1)
        elif image_representation=='tivit_grayscale':
            rgb=np.repeat(np.clip(matrices[ch],0,1)[...,None],3,axis=-1)
            panel=Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((width,height),Image.Resampling.NEAREST)
        else:
            rgb=colormaps['viridis'](np.clip(matrices[ch],0,1))[...,:3]
            rgb[~mask[ch]]=.5
            panel=Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize((width,height),Image.Resampling.NEAREST)
        canvas.paste(panel,(left,top))
    return torch.from_numpy(np.array(canvas,copy=True)).permute(2,0,1).float()/255
