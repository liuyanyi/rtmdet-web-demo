import streamlit as st
import torch
from model import RTMDetModel2Img, load_rtmdet_models
from pinferencia import Server

task_name = dict(
    det='RTMDet',
    ins='RTMDet-Ins',
    rot='RTMDet-R'
)

tasks = ['det', 'ins', 'rot']
sizes = ['tiny', ]

res = load_rtmdet_models(tasks, sizes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(ttl=6400)
def build_model(model_name, meta, device):
    return RTMDetModel2Img(model_name, meta, device)


service = Server()
for t in tasks:
    for s in sizes:
        model_name = task_name[t]+" "+s
        service.register(
            model_name=task_name[t]+" "+s,
            model=build_model(model_name, res[t][s], device),
            entrypoint="predict")

# model = MyModel()
