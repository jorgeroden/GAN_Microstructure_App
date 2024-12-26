DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NZ = 100
NGF = 16
MODEL_PATHS = {
    "martensite": "models/netG_martensite_512.pt",
    "pearlite": "models/netG_pearlite_512.pt",
}
PORT = 8089