import torch.nn as nn

# Define constants for the generator architecture
nz = 100  # Size of the latent vector
ngf = 16  # Size of feature maps in generator
filterSize = 4  # Kernel size for convolutions
nc = 3  # Number of channels in the generated images (e.g., 3 for RGB)

class Generator(nn.Module):
    """
    Generator model for GAN-based image generation using configurable parameters.
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 64, filterSize, 1, 0, bias=False),  # Shape: N x (ngf*64) x 4 x 4
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 64, ngf * 32, filterSize, 2, 1, bias=False),  # Shape: N x (ngf*32) x 8 x 8
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 32, ngf * 16, filterSize, 2, 1, bias=False),  # Shape: N x (ngf*16) x 16 x 16
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, filterSize, 2, 1, bias=False),  # Shape: N x (ngf*8) x 32 x 32
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, filterSize, 2, 1, bias=False),  # Shape: N x (ngf*4) x 64 x 64
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, filterSize, 2, 1, bias=False),  # Shape: N x (ngf*2) x 128 x 128
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, filterSize, 2, 1, bias=False),  # Shape: N x ngf x 256 x 256
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, filterSize, 2, 1, bias=False),  # Shape: N x nc x 512 x 512
            nn.Tanh()  # Normalize output to [-1, 1]
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        """
        Forward pass of the generator.
        Args:
            input (torch.Tensor): Latent vector (noise).
        Returns:
            torch.Tensor: Generated image tensor.
        """
        return self.main(input)

