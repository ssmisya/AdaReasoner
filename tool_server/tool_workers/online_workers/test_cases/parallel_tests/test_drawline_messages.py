"""Send parallel drawline test messages."""
import argparse
import json
import time
from io import BytesIO
import cv2
import sys
import numpy as np

import requests
from PIL import Image
import base64

import torch
import torchvision.transforms.functional as F

from tool_server.utils.utils import *
from concurrent.futures import ThreadPoolExecutor
import threading
import os

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # resize if needed
    w, h = img.size
    if max(h, w) > 800:
        if h > w:
            new_h = 800
            new_w = int(w * 800 / h)
        else:
            new_w = 800
            new_h = int(h * 800 / w)
        img = F.resize(img, (new_h, new_w))
    return img

def encode(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    return img_b64_str

def test_parallel_drawline():
    print("Testing parallel drawline requests...")
    
    # Configuration for parallel tests
    num_threads = args.num_threads
    model_name = args.model_name
    
    # Get worker address
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        controller_addr = args.controller_address
        ret = requests.post(controller_addr + "/list_models")
        print(f"list_models: {ret.json()}")
        models = ret.json()["models"]
        models.sort()
        print(f"Models: {models}")
        
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
            "param": args.caption,
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
            
            if response.status_code == 200:
                res = response.json()
                # Save the result with thread_id to distinguish between responses
                if "edited_image" in res:
                    image_base64 = res["edited_image"]
                    image = base64_to_pil(image_base64)
                    image.save(f"drawline_image_{thread_id}.jpg")
                
                print(f"Thread {thread_id}: Success in {elapsed:.3f}s")
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
        "--controller-address", type=str, default="http://SH-IDCA1404-10-140-54-2:20001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='DrawHorizontalLineByY')

    # model parameters
    parser.add_argument(
        "--caption", type=str, default="<point x=\"51.5\" y=\"82.0\" alt=\"x = 2017\">x = 2017</point>"
    )
    parser.add_argument(
        "--image_path", type=str, default="/mnt/petrelfs/songmingyang/code/tools/test_imgs/roxy.jpeg"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3,
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25,
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    parser.add_argument(
        "--num-threads", type=int, default=100, help="Number of parallel requests"
    )
    args = parser.parse_args()
    args.send_image = True

    test_parallel_drawline()
