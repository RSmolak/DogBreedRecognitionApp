import numpy as np
from getModel import get_model
from starlette.responses import HTMLResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import torchvision
import torch
from fastapi import FastAPI, File
from starlette.requests import Request
from fastapi.templating import Jinja2Templates


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model('./model1.pt')
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
])
model.to(device)
trainset = torchvision.datasets.ImageFolder("dataset/train")


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def upload_image(request: Request):
    return templates.TemplateResponse("uploadview.html", {"request": request})


# @app.post("/detect")
# async def handle_form(request: Request, file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")
#     detect_res = json.loads(detect_res)
#     number = str(len(detect_res))
#     return {number}

    # input_image = get_image_from_bytes(file)
    # results = model(input_image)
    # detect_res = results.pandas().xyxy[0].to_json(orient="records")
    # detect_res = json.loads(detect_res)
    # number = str(len(detect_res))
    # results.render()
    # for img in results.imgs:
    #     bytes_io = io.BytesIO()
    #     img_base64 = Image.fromarray(img)
    #     img_base64.save(bytes_io, format="jpeg")
    # return templates.TemplateResponse("detect.html", context={"request": request}),


    # input_image = get_image_from_bytes(file)
    # results = model(input_image)
    # results.render()
    # for img in results.imgs:
    #     bytes_io = io.BytesIO()
    #     img_base64 = Image.fromarray(img)
    #     img_base64.save(bytes_io, format="jpeg")
    # return Response(content=bytes_io.getvalue(), media_type="image/jpeg")

#
# @app.post("/img-to-json")
# async def detect_people_return_json_result(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")
#     detect_res = json.loads(detect_res)
#     return {"result": detect_res}
#
#
# @app.post("/img-to-img")
# async def detect_people_return_base64_img(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     results.render()
#     for img in results.imgs:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
#
#
@app.post("/img-to-string")
async def recognise_breed(file: bytes = File(...)):
    model.eval()
    img = Image.open(io.BytesIO(file))
    img = transforms(img)
    pred = int(np.squeeze(model(img.unsqueeze(0).to(device)).data.max(1, keepdim=True)[1].cpu().numpy()))
    pred = trainset.classes[pred]
    preds = torch.from_numpy(np.squeeze(model(img.unsqueeze(0).to(device)).data.cpu().numpy()))
    #top_preds = torch.topk(torch.exp(preds), 5)
    #top_preds = dict(zip([trainset.classes[i] for i in top_preds.indices],
    #                     [f"{round(float(i) * 100, 2)}%" for i in top_preds.values]))
    return {str(pred)}
#
#
# @app.post("/file-to-number")
# async def detect_people_return_number(file: UploadFile = File(...)):
#     with open(file.filename, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     results = model(file.filename)
#     detect_res = results.pandas().xyxy[0].to_json(orient="records")
#     detect_res = json.loads(detect_res)
#     number = str(len(detect_res))
#     os.remove(file.filename)
#     return {number}


# @app.post("/file-to-string")
# async def recognise_breed(file: UploadFile = File(...)):
#     with open(file.filename, 'rb') as buffer:
#         buffer.write(file.read())
#     model.eval()
#     Image.open(io.BytesIO(file.))
#     img = transforms(file.filename)
#     pred = int(np.squeeze(model(img.unsqueeze(0).to(device)).data.max(1, keepdim=True)[1].cpu().numpy()))
#     pred = trainset.classes[pred]
#     preds = torch.from_numpy(np.squeeze(model(img.unsqueeze(0).to(device)).data.cpu().numpy()))
#     top_preds = torch.topk(torch.exp(preds), 5)
#     top_preds = dict(zip([trainset.classes[i] for i in top_preds.indices],
#                          [f"{round(float(i) * 100, 2)}%" for i in top_preds.values]))
#
#     results = model(file.filename)
#     recognise = results.pandas().xyxy[0].to_json(orient="records")
#     recognise = json.loads(recognise)
#     number = str(len(recognise))
#     os.remove(file.filename)
#     return {str(recognise)}
