from fastapi import FastAPI
from starlette.responses import Response
from backend.utils import load_model, normalize_tensor
import torch
from backend.models import Generator
import io
from PIL import Image
from torchvision import transforms as Transforms


app = FastAPI()

@app.get("/martensite")
def generate_martensite():
    """
    Generate an image of martensite using the trained GAN model.
    Returns:
        Response: Generated image in PNG format.
    """
    
    model_path = "models/netG_martensite_512_280_epochs_16filters_batchsize4_09APR24.pt"
    generator = load_model(model_path, "cpu")
  
    b_size = 1

    nz = 100
    noise4image = torch.randn(b_size, nz, 1, 1, device="cpu")

  
    image_fake = generator(noise4image)
    img = image_fake[0]  # Tomar la primera imagen del batch

  
    img = normalize_tensor(img)
    img = img.detach().cpu()  # Mover el tensor a la CPU para procesamiento

  
    transform = Transforms.Compose([Transforms.ToPILImage()])
    generated_image = transform(img.clamp(min=-1, max=1))

  
    bytes_io = io.BytesIO()
    generated_image.save(bytes_io, "PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
    

@app.get("/pearlite")
def generate_pearlite():
    """
    Generate an image of pearlite using the trained GAN model.
    Returns:
        Response: Generated image in PNG format.
    """
    

    model_path = "models/netG_pearlite_512_300_epochs_16filters_31MAR24_minibatch8.pt"
    generator = load_model(model_path, "cpu")
    
 
    b_size = 1

    nz = 100
    noise4image = torch.randn(b_size, nz, 1, 1, device="cpu")

  
    image_fake = generator(noise4image)
    img = image_fake[0]  # Tomar la primera imagen del batch

  
    img = normalize_tensor(img)
    img = img.detach().cpu()  # Mover el tensor a la CPU para procesamiento

  
    transform = Transforms.Compose([Transforms.ToPILImage()])
    generated_image = transform(img.clamp(min=-1, max=1))

  
    bytes_io = io.BytesIO()
    generated_image.save(bytes_io, "PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
    

