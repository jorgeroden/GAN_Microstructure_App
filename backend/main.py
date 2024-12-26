from fastapi import FastAPI
from starlette.responses import Response
from utils import load_model, normalize_tensor
import torch
from models import Generator
import io
from PIL import Image

app = FastAPI()

@app.get("/martensite")
def generate_martensite():

    model_path = "models/netG_martensite_512_280_epochs_16filters_batchsize4_09APR24.pt"
    generator = load_model(model_path, "cpu")
    noise = torch.randn(1, 100, 1, 1, device="cpu")
    image_tensor = generator(noise).squeeze(0).detach().cpu()
    image_tensor = normalize_tensor(image_tensor)
    image = Image.fromarray((image_tensor.numpy() * 255).astype("uint8"))
    bytes_io = io.BytesIO()
    image.save(bytes_io, "PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")

@app.get("/pearlite")
def generate_pearlite():
    model_path = "models/netG_pearlite_512_300_epochs_16filters_31MAR24_minibatch8.pt"
    generator = load_model(model_path, "cpu")
    noise = torch.randn(1, 100, 1, 1, device="cpu")
    image_tensor = generator(noise).squeeze(0).detach().cpu()
    image_tensor = normalize_tensor(image_tensor)
    image = Image.fromarray((image_tensor.numpy() * 255).astype("uint8"))
    bytes_io = io.BytesIO()
    image.save(bytes_io, "PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
