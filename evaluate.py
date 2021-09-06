import gradio as gr
import numpy as np
from numpy.core.fromnumeric import resize
import timm
import torch
from PIL import Image


model_name = "model_mark1.pth"
# model_official = timm.create_model(model_name, pretrained=True)
model = torch.load('/home/charchit/Desktop/MDETR/model_mark1.pth')
model.eval()

imagenet_labels = dict(enumerate(open("classes.txt")))

def func(img):

    # using gradio for web app
    # img = Image.open("/home/charchit/Desktop/dog.jpg")
    # img = img.resize((224,224))
    # img = np.array(img)/128
    img = img/128
    # img = img[...,0:3]
    inp = torch.from_numpy(img).permute(2,0,1).to(torch.float32)
    imp = inp[np.newaxis,...]
    logits = model(imp)
    probs = torch.nn.functional.softmax(logits,dim=-1)

    top_probs, top_ixs = probs[0].topk(1)

    for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
        ix = ix_.item()
        prob = prob_.item()
        cls = imagenet_labels[ix].strip()
        return (f"{i}: {cls:<45} --- {prob:.4f}")

iface = gr.Interface(func,gr.inputs.Image(shape=(224,224)), "text")
iface.launch()

