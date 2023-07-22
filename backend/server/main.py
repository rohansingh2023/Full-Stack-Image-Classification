import uvicorn
from fastapi import FastAPI,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from torch.autograd import Variable
from PIL import Image
from modelConfig import NNArch, transform, classes
import io

# Initialize API Server
app = FastAPI(
    title="Intel Classification",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Loading model
checkpoint = torch.load('../intelClass.pth', map_location=torch.device('cpu'))
model = NNArch()
model.load_state_dict(checkpoint)
model.eval()

# Specifying Request type
class Req(BaseModel):
    image: str

@app.get('/')
def home():
    return {"Iron":"Man"}

@app.post('/predict')
async def prediction(file: UploadFile = File(...)):
    img_bytes = file.file.read()
    image = Image.open(io.BytesIO(img_bytes))
    image_tensor=transform(image).float()
    image_tensor=image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
    input=Variable(image_tensor)
    output=model(input)
    index=output.data.numpy().argmax()
    pred=classes[index]
    return pred

if __name__ == '__main__':
    uvicorn.run("main:app", port=8080,
                reload=True
                )