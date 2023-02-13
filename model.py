import base64
import importlib
from io import BytesIO
from typing import List, Optional, Union

import numpy as np
import streamlit as st
import torch
from mim.commands.search import get_model_info
from mmcv.image import imread
from mmcv.transforms import Compose
from mmdet.apis.inference import init_detector
from mmdet.registry import VISUALIZERS
from mmdet.utils.misc import get_test_pipeline_cfg
from mmengine.hub import get_config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from utils import hbox2qbox, rbox2qbox

Opt_Input_Type = Optional[Union[str, List[str]]]


def filter_model_name(key: str, task: str, size: str):
    if task == 'det':
        return key.split('_')[0] == 'rtmdet' and key.split('_')[1] == size
    elif task == 'ins':
        return key.split('_')[0] == 'rtmdet-ins' and key.split('_')[1] == size
    elif task == 'rot':
        # TODO not safe
        return key.split('_')[2].split('-')[0] == size
    else:
        raise NotImplementedError


@st.cache_data(ttl=6400)
def load_rtmdet_models(tasks: Opt_Input_Type = ['det', 'ins', 'rot'],
                       sizes: Opt_Input_Type = ['tiny', 's', 'm', 'l', 'x']):
    if isinstance(tasks, str):
        tasks = [tasks, ]
    if isinstance(sizes, str):
        sizes = [sizes, ]
    mmdet_models = None
    mmrotate_models = None
    if 'det' in tasks or 'ins' in tasks:
        mmdet_models = get_model_info(
            'mmdet', models=['rtmdet', ], to_dict=True)
    if 'rot' in tasks:
        mmrotate_models = get_model_info(
            'mmrotate', models=['rotated_rtmdet', ], to_dict=True)
    model_dicts = dict()
    for task in tasks:
        res_dict = dict()
        if task == 'rot':
            models = mmrotate_models
            scope = 'mmrotate'
        else:
            models = mmdet_models
            scope = 'mmdet'
        for size in sizes:
            model_key = list(
                filter(lambda k: filter_model_name(k, task, size), models.keys()))
            if len(model_key) == 0:
                print('Not Exist')
                pass
            else:
                model_key = model_key[0]
                res_dict[size] = dict(
                    scope=scope,
                    weight=models[model_key]['weight'],
                    cfg=scope + "::" + models[model_key]['config'][8:]
                )
        model_dicts[task] = res_dict
    return model_dicts


class RTMDetModel:

    def __init__(
        self,
        name: str,
        meta: dict,
        device: torch.device
    ):
        cfg = get_config(meta['cfg'])

        models_module = importlib.import_module(f'{meta["scope"]}.utils')
        models_module.register_all_modules()  # type: ignore

        self.model = init_detector(cfg, meta['weight'], device=device)

        # self.model = get_model(meta['cfg'], pretrained=True).to(device)
        # self.model.cfg = get_config(meta['cfg'])

        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)

        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        self.test_pipeline = Compose(test_pipeline)

        revert_sync_batchnorm(self.model)

        vis_cfg = cfg.visualizer.copy()
        vis_cfg.name = name

        self.visualizer = VISUALIZERS.build(vis_cfg)
        # Handle mmrotate checkpoint error
        if not hasattr(self.model, 'dataset_meta') and meta["scope"] == 'mmrotate':
            self.model.dataset_meta = {
                'classes':
                ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                 'harbor', 'swimming-pool', 'helicopter'),
                # palette is a list of color tuples, which is used for visualization.
                'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                            (138, 43, 226), (255, 128, 0), (255, 0, 255),
                            (0, 255, 255), (255, 193, 193), (0, 51, 153),
                            (255, 250, 205), (0, 139, 139), (255, 255, 0),
                            (147, 116, 116), (0, 0, 255)]
            }
        self.visualizer.dataset_meta = self.model.dataset_meta

    def predict(self, img_str: str) -> list:
        img = Image.open(BytesIO(base64.b64decode(img_str)))
        img = imread(np.array(img))
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = self.test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = self.model.test_step(data_)[0]

        return self.postprocess(results, img)

    def postprocess(self, data, img=None) -> list:
        data = data.pred_instances
        data = data.to('cpu')

        # res_data = dict(
        #     score=pp.scores.numpy().tolist(),
        #     box=pp.bboxes.numpy().tolist(),
        #     label=pp.labels.numpy().tolist(),
        # )
        result_list = []
        for i, (score, bbox, label) in enumerate(zip(data.scores, data.bboxes, data.labels)):
            result = dict()
            result['score'] = score.numpy().item()
            if bbox.size(-1) == 4:
                result['bbox'] = hbox2qbox(bbox).numpy().tolist()
            elif bbox.size(-1) == 5:
                result['bbox'] = rbox2qbox(bbox).numpy().tolist()
            else:
                result['bbox'] = bbox.numpy().tolist()
            result['label'] = label.numpy().item()
            result_list.append(result)

        return result_list


class RTMDetModel2Img(RTMDetModel):

    def predict(self, img_str: str) -> str:
        return super().predict(img_str)

    def postprocess(self, data, img) -> str:
        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        self.visualizer.add_datasample(
            'result',
            img,
            data_sample=data,
            draw_gt=False,
            show=None,
            wait_time=0,
            out_file=None,
            pred_score_thr=0.3)
        res_img = self.visualizer.get_image()

        pil_img = Image.fromarray(res_img)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")


# def main():
#     res = load_rtmdet_models()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     ccc = RTMDetModel(res['ins']['tiny'], device)
#     # res_data = ccc.predict('/workspace/dfc2023/mmrotate/demo/demo.jpg')

#     print(res_data)


# if __name__ == '__main__':
#     main()
