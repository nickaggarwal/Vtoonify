# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from transformers import CLIPTokenizer
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
import requests as req
import torch
import torch.nn as nn
import numpy as np
import dlib
import cv2
import torch.nn.functional as F
from torchvision import transforms
import gc
import huggingface_hub
import os
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL

from model_package.vtoonify import VToonify
from model_package.bisenet.model import BiSeNet
from model_package.encoder.align_all_parallel import align_face
from util import load_psp_standalone, get_video_crop_parameter


MODEL_REPO = 'PKUWilliamYang/VToonify'

class TritonPythonModel:

    def initialize(self, args):
        self.device = "cuda:0"
        self.landmarkpredictor = self._create_dlib_landmark_model()
        self.parsingpredictor = self._create_parsing_model()
        self.pspencoder = self._load_encoder()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.vtoonify, self.exstyle = self._load_default_model()
        self.color_transfer = False
        self.style_name = 'cartoon1'
        self.video_limit_cpu = 100
        self.video_limit_gpu = 300



    @staticmethod
    def _create_dlib_landmark_model():
        return dlib.shape_predictor(huggingface_hub.hf_hub_download(MODEL_REPO,
                                                                    'models/shape_predictor_68_face_landmarks.dat'))

    def _create_parsing_model(self):
        parsingpredictor = BiSeNet(n_classes=19)
        parsingpredictor.load_state_dict(
            torch.load(huggingface_hub.hf_hub_download(MODEL_REPO, 'models/faceparsing.pth'),
                       map_location=lambda storage, loc: storage))
        parsingpredictor.to("cuda:0").eval()
        return parsingpredictor

    def _load_encoder(self):
        style_encoder_path = huggingface_hub.hf_hub_download(MODEL_REPO, 'models/encoder.pt')
        return load_psp_standalone(style_encoder_path, "cuda:0")

    def _load_default_model(self):
        vtoonify = VToonify(backbone='dualstylegan')
        vtoonify.load_state_dict(torch.load(huggingface_hub.hf_hub_download(MODEL_REPO,
                                                                            'models/vtoonify_d_cartoon/vtoonify_s026_d0.5.pt'),
                                            map_location=lambda storage, loc: storage)['g_ema'])
        vtoonify.to("cuda:0")
        tmp = np.load(huggingface_hub.hf_hub_download(MODEL_REPO, 'models/vtoonify_d_cartoon/exstyle_code.npy'),
                      allow_pickle=True).item()
        exstyle = torch.tensor(tmp[list(tmp.keys())[26]]).to("cuda:0")
        with torch.no_grad():
            exstyle = vtoonify.zplus2wplus(exstyle)
        return vtoonify, exstyle

    def load_model(self, style_type: str):
        if 'illustration' in style_type:
            self.color_transfer = True
        else:
            self.color_transfer = False
        if style_type not in self.style_types.keys():
            return None, 'Oops, wrong Style Type. Please select a valid model.'
        self.style_name = style_type
        model_path, ind = self.style_types[style_type]
        style_path = os.path.join('models', os.path.dirname(model_path), 'exstyle_code.npy')
        self.vtoonify.load_state_dict(torch.load(huggingface_hub.hf_hub_download(MODEL_REPO, 'models/' + model_path),
                                                 map_location=lambda storage, loc: storage)['g_ema'])
        tmp = np.load(huggingface_hub.hf_hub_download(MODEL_REPO, style_path), allow_pickle=True).item()
        exstyle = torch.tensor(tmp[list(tmp.keys())[ind]]).to(self.device)
        with torch.no_grad():
            exstyle = self.vtoonify.zplus2wplus(exstyle)
        return exstyle, 'Model of %s loaded.' % (style_type)

    def detect_and_align(self, frame, top, bottom, left, right, return_para=False):
        message = 'Error: no face detected! Please retry or change the photo.'
        paras = get_video_crop_parameter(frame, self.landmarkpredictor, [left, right, top, bottom])
        instyle = None
        h, w, scale = 0, 0, 0
        if paras is not None:
            h, w, top, bottom, left, right, scale = paras
            H, W = int(bottom - top), int(right - left)
            # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
            kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
            with torch.no_grad():
                I = align_face(frame, self.landmarkpredictor)
                if I is not None:
                    I = self.transform(I).unsqueeze(dim=0).to(self.device)
                    instyle = self.pspencoder(I)
                    instyle = self.vtoonify.zplus2wplus(instyle)
                    message = 'Successfully rescale the frame to (%d, %d)' % (bottom - top, right - left)
                else:
                    frame = np.zeros((256, 256, 3), np.uint8)
        else:
            frame = np.zeros((256, 256, 3), np.uint8)
        if return_para:
            return frame, instyle, message, w, h, top, bottom, left, right, scale
        return frame, instyle, message

    # @torch.inference_mode()
    def detect_and_align_image(self, image , top: int, bottom: int, left: int, right: int
                               ):
        if image is None:
            return np.zeros((256, 256, 3), np.uint8), None, 'Error: fail to load the image.'
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return self.detect_and_align(frame, top, bottom, left, right)

    def execute(self, requests):

        responses = []
        for request in requests:

            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = inp.as_numpy()[0][0].decode()

            inp_image = pb_utils.get_input_tensor_by_name(request, "image")
            image_array = inp_image.as_numpy()[0][0]
            image_array = image_array.astype(np.uint8)
            print(image_array.shape)
            pil_image = PIL.Image.fromarray(image_array).convert("RGB")
            open_cv_image = np.array(pil_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()

            aligned_face, instyle, style_type = self.detect_and_align_image(open_cv_image, 200, 200, 200, 200)
            
            # Load the model based of style
            exstyle = pb_utils.get_input_tensor_by_name(request, "style")
            load_model(exstyle)

            style_degree: float = 0.5
            style_type: str = input_text

            if instyle is None or aligned_face is None:
                return np.zeros((256, 256, 3), np.uint8), 'Opps, something wrong with the input. Please go to Step 2 and Rescale Image/First Frame again.'
            if exstyle is None:
                return np.zeros((256, 256, 3),  np.uint8), 'Opps, something wrong with the style type. Please go to Step 1 and load model again.'

            with torch.no_grad():
                if self.color_transfer:
                    s_w = exstyle
                else:
                    s_w = instyle.clone()
                    s_w[:, :7] = exstyle[:, :7]

                x = self.transform(aligned_face).unsqueeze(dim=0).to(self.device)
                x_p = F.interpolate(
                    self.parsingpredictor(2 * (F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[
                        0],
                    scale_factor=0.5, recompute_scale_factor=False).detach()
                inputs = torch.cat((x, x_p / 16.), dim=1)
                y_tilde = self.vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=style_degree)
                y_tilde = torch.clamp(y_tilde, -1, 1)
                print('*** Toonify %dx%d image with style of %s' % (y_tilde.shape[2], y_tilde.shape[3], style_type))
                image_arr = (y_tilde[0].cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5


            # Sending results
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    np.array(image_arr, dtype=np.float32),
                )
            ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        self.pipe = None

