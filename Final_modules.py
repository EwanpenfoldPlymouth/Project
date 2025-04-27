import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA:
    """
    Exponential Moving Average (EMA) helper for model parameters.
    """
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    @torch.no_grad()
    def update_model_average(self, ma_model, current_model):
        """
        Updates the EMA model's parameters using in-place fixed-point operations.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # In-place update: new_avg = beta * old_avg + (1 - beta) * new_val
            ma_params.data.mul_(self.beta).add_(current_params.data * (1 - self.beta))
            
    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Performs an EMA update once the training step has passed a threshold.
        Before this threshold, the EMA model is simply reset to the current model.
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the EMA model's parameters to match the current model.
        """
        ema_model.load_state_dict(model.state_dict())


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding module that transforms a sinusoidal positional encoded time input
    into a high-dimensional embedding space.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, t):
        """
        Forward pass to obtain time embeddings.
        Args:
            t (Tensor): Time embeddings of shape (B, emb_dim).
        Returns:
            Tensor: Learned time embeddings of shape (B, emb_dim).
        """
        return self.linear2(self.act(self.linear1(t)))


class SelfAttention(nn.Module):
    """
    Self-attention layer for capturing long-range dependencies in feature maps.
    """
    def __init__(self, channels, size=None):
        """
        Args:
            channels (int): Number of channels in the input feature map.
            size (int, optional): Spatial size (H or W) of the feature map. Currently unused.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        # The 'size' parameter is kept for potential future use (e.g. for relative position biases).
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map of shape (B, channels, H, W).
        Returns:
            Tensor: Output feature map of shape (B, channels, H, W).
        """
        B, C, H, W = x.shape
        # Reshape for multihead attention: (B, H*W, channels)
        x_reshaped = x.view(B, C, H * W).transpose(1, 2)
        x_ln = self.ln(x_reshaped)
        attn_output, _ = self.mha(x_ln, x_ln, x_ln)
        # Residual connections with feedforward network
        attn_output = attn_output + x_reshaped
        ff_output = self.ff_self(attn_output) + attn_output
        # Reshape back to (B, channels, H, W)
        return ff_output.transpose(1, 2).view(B, C, H, W)


class DoubleConv(nn.Module):
    """
    A block with two consecutive convolutional layers,
    each followed by GroupNorm and GELU activation.
    Optionally adds a residual connection.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, num_groups=8):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
        )

    def forward(self, x):
        if self.residual:
            # Residual connection after the two-layer convolutional block.
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Down-sampling block: applies max pooling, followed by two DoubleConv layers,
    and adds time conditioning from a learned embedding.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256, num_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True, num_groups=num_groups),
            DoubleConv(in_channels, out_channels, num_groups=num_groups),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        Args:
            x (Tensor): Input image feature map.
            t (Tensor): Time embedding tensor of shape (B, emb_dim).
        Returns:
            Tensor: Output feature map with time conditioning added.
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None]  # Reshape to match spatial dims.
        return x + emb.expand_as(x)


class Up(nn.Module):
    """
    Up-sampling block: upsamples the input, concatenates it with skip connections,
    applies convolution, and adds time conditioning.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256, num_groups=8):
        """
        Args:
            in_channels (int): Number of channels after concatenation of features.
            out_channels (int): Desired output channels.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, num_groups=num_groups),
            DoubleConv(in_channels, out_channels, in_channels // 2, num_groups=num_groups),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        Args:
            x (Tensor): Input feature map to upsample.
            skip_x (Tensor): Corresponding feature map from encoder (skip connection).
            t (Tensor): Time embedding tensor.
        Returns:
            Tensor: Output feature map after upsampling and time conditioning.
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None]
        return x + emb.expand_as(x)


class UNet(nn.Module):
    """
    U-Net architecture for denoising diffusion models with learned time conditioning.
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # Encoding path
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention(256)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoding path
        # Note: for up1 the concatenated channels (from upsampled bottleneck and skip connection)
        # equal 256 + 256 = 512.
        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128)
        # For up2: 128 (from up1) + 128 (skip from down1) = 256.
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64)
        # For up3: 64 (from up2) + 64 (skip from inc) = 128.
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        
        # Time embedding: first, compute a sinusoidal positional encoding, then learn a projection.
        self.time_mlp = TimeEmbedding(time_dim)

    def pos_encoding(self, t, channels):
        """
        Computes the sinusoidal positional embedding for a given time tensor.
        Args:
            t (Tensor): Tensor of shape (B, 1) containing scalar time values.
            channels (int): Output embedding dimension.
        Returns:
            Tensor: Positional encodings of shape (B, channels).
        """
        t = t.to(self.device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        """
        Forward pass of the UNet.
        Args:
            x (Tensor): Input image tensor of shape (B, c_in, H, W).
            t (Tensor): Time steps tensor of shape (B,).
        Returns:
            Tensor: Output image tensor of shape (B, c_out, H, W).
        """
        # Prepare time embeddings:
        t = t.unsqueeze(-1).type(torch.float)  # Shape: (B, 1)
        t = self.pos_encoding(t, self.time_dim)  # Shape: (B, time_dim)
        t = self.time_mlp(t)  # Learned projection

        # Down-sampling path
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x_bottleneck = self.bot1(x4)
        x_bottleneck = self.bot2(x_bottleneck)
        x_bottleneck = self.bot3(x_bottleneck)

        # Up-sampling path with skip connections
        x = self.up1(x_bottleneck, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
