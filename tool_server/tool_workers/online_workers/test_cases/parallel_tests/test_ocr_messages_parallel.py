"""Send a test message."""
import argparse
import json
import time
from io import BytesIO
import cv2
import sys
import numpy as np


import requests
import base64

import torch
import torchvision.transforms.functional as F

import uuid
import os
import re
import io

from PIL import Image, ImageDraw
from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
import matplotlib.pyplot as plt

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

# from tool_server.utils.cogcom.models.cogcom_model import CogCoMModel
# from tool_server.utils.cogcom.utils import chat
# from tool_server.utils.cogcom.utils import get_image_processor, llama2_tokenizer, llama2_text_processor_inference
import threading
from concurrent.futures import ThreadPoolExecutor


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # import ipdb; ipdb.set_trace()
    # resize if needed
    w, h = img.size
    if max(h, w) > 800:
        if h > w:
            new_h = 800
            new_w = int(w * 800 / h)
        else:
            new_w = 800
            new_h = int(h * 800 / w)
        # import ipdb; ipdb.set_trace()
        img = F.resize(img, (new_h, new_w))
    return img

def encode(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str


# def main():
#     model_name = args.model_name

#     if args.worker_address:
#         worker_addr = args.worker_address
#     else:
#         controller_addr = args.controller_address
#         # ret = requests.post(controller_addr + "/refresh_all_workers")
#         ret = requests.post(controller_addr + "/list_models")
#         print(f"list_models: {ret.json()}")
#         models = ret.json()["models"]
#         models.sort()
#         print(f"Models: {models}")

#         ret = requests.post(
#             controller_addr + "/get_worker_address", json={"model": model_name}
#         )
#         worker_addr = ret.json()["address"]
#         print(f"worker_addr: {worker_addr}")

#     if worker_addr == "":
#         print(f"No available workers for {model_name}")
#         return

#     headers = {"User-Agent": "FastChat Client"}
#     if args.send_image:
#         img = load_image(args.image_path)
#         img_arg = encode(img)
#     else:
#         img_arg = args.image_path
#     datas = {
#         "model": model_name,
#         "image": img_arg,
#     }
#     tic = time.time()
#     response = requests.post(
#         worker_addr + "/worker_generate",
#         headers=headers,
#         json=datas,
#     )
#     toc = time.time()
#     print(f"Time: {toc - tic:.3f}s")

#     print("detection result:")
#     # print(response)
#     print(response.json())
#     # response is 'Response' with :
#     # ['_content', '_content_consumed', '_next', 'status_code', 'headers', 'raw', 'url', 'encoding', 'history', 'reason', 'cookies', 'elapsed', 'request', 'connection', '__module__', '__doc__', '__attrs__', '__init__', '__enter__', '__exit__', '__getstate__', '__setstate__', '__repr__', '__bool__', '__nonzero__', '__iter__', 'ok', 'is_redirect', 'is_permanent_redirect', 'next', 'apparent_encoding', 'iter_content', 'iter_lines', 'content', 'text', 'json', 'links', 'raise_for_status', 'close', '__dict__', '__weakref__', '__hash__', '__str__', '__getattribute__', '__setattr__', '__delattr__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__new__', '__reduce_ex__', '__reduce__', '__subclasshook__', '__init_subclass__', '__format__', '__sizeof__', '__dir__', '__class__']

#     # visualize
#     res = response.json()
#     print(f"response: {res['text']}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # worker parameters
#     parser.add_argument(
#         "--controller-address", type=str, default="http://SH-IDCA1404-10-140-54-119:20001"
#     )
#     parser.add_argument("--worker-address", type=str)
#     parser.add_argument("--model-name", type=str, default='ocr')

#     # model parameters
#     parser.add_argument(
#         "--send_image", action="store_true",
#     )
#     parser.add_argument(
#         "--image_path", type=str, default="/mnt/petrelfs/haoyunzhuo/mmtool/Tool-Factory/tool_server/tool_workers/online_workers/test_cases/subplot_0.png"
#     )
#     args = parser.parse_args()

#     main()
def test_parallel_ocr():

    print("Testing parallel OCR requests...")
    
    # Configuration for parallel tests
    num_threads = 10  # Number of parallel requests
    model_name = args.model_name
    
    # Get worker address
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/list_models")
        print(f"list_models: {ret.json()}")
        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        print(f"worker_addr: {worker_addr}")

    if worker_addr == "":
        print(f"No available workers for {model_name}")
        return

    # Prepare image data
    if args.send_image:
        img = load_image(args.image_path)
        img_arg = encode(img)
    else:
        img_arg = args.image_path
    
    # Request function for each thread
    def make_request(thread_id):
        headers = {"User-Agent": f"FastChat Client Thread-{thread_id}"}
        datas = {
            "model": model_name,
            "image": img_arg,
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                worker_addr + "/worker_generate",
                headers=headers,
                json=datas,
            )
            elapsed = time.time() - start_time
            
            # Check if response is valid
            if response.status_code == 200:
                result = response.json()
                
                print(f"Thread {thread_id}: {result} Success in {elapsed:.3f}s")
                return True, elapsed, thread_id
            else:
                print(f"Thread {thread_id}: Failed with status {response.status_code} in {elapsed:.3f}s")
                return False, elapsed, thread_id
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Thread {thread_id}: Exception {str(e)} in {elapsed:.3f}s")
            return False, elapsed, thread_id
    
    # Execute parallel requests
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_threads)]
        for future in futures:
            results.append(future.result())
    
    # Analyze results
    success_count = sum(1 for r in results if r[0])
    if success_count == num_threads:
        print(f"✅ All {num_threads} parallel requests succeeded")
    else:
        print(f"❌ Only {success_count}/{num_threads} requests succeeded")
    
    # Calculate statistics
    times = [r[1] for r in results]
    print(f"Average response time: {sum(times)/len(times):.3f}s")
    print(f"Min response time: {min(times):.3f}s")
    print(f"Max response time: {max(times):.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://SH-IDCA1404-10-140-54-1:20001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='OCR')

    # model parameters
    parser.add_argument(
        "--send_image", action="store_true",
    )
    parser.add_argument(
        "--image_path", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers/online_workers/test_cases/subplot_0.png"
    )
    parser.add_argument(
        "--test-parallel", action="store_true", help="Test parallel OCR requests"
    )
    args = parser.parse_args()
    args.send_image=True

    test_parallel_ocr()