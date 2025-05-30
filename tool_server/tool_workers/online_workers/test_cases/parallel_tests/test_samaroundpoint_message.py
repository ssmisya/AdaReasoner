"""Send test messages in parallel."""
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
import threading
from concurrent.futures import ThreadPoolExecutor


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


def single_request(thread_id, worker_addr, model_name, img_arg, point_param=None):
    headers = {"User-Agent": f"FastChat Client Thread-{thread_id}"}
    datas = {
        "model": model_name,
        "param": point_param,
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
            
            print(f"Thread {thread_id}: Success in {elapsed:.3f}s")
            
            # Save the output image if available
            if "edited_image" in result:
                image = base64_to_pil(result["edited_image"])
                image.save(f"truck_segmented_test_{thread_id}.jpg")
                
            return True, elapsed, thread_id, result
        else:
            print(f"Thread {thread_id}: Failed with status {response.status_code} in {elapsed:.3f}s")
            return False, elapsed, thread_id, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Thread {thread_id}: Exception {str(e)} in {elapsed:.3f}s")
        return False, elapsed, thread_id, None


def test_parallel_segment():
    print("Testing parallel segment region around point requests...")
    
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
    
    # Execute parallel requests
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(single_request, i, worker_addr, model_name, img_arg, args.point_response) 
                  for i in range(num_threads)]
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


def main():
    if args.parallel:
        test_parallel_segment()
    else:
        # Original single-request logic
        model_name = args.model_name

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

        headers = {"User-Agent": "FastChat Client"}
        if args.send_image:
            img = load_image(args.image_path)
            img_arg = encode(img)
        else:
            img_arg = args.image_path
        datas = {
            "model": model_name,
            "param": args.point_response,
            "image": img_arg,
        }
        tic = time.time()
        response = requests.post(
            worker_addr + "/worker_generate",
            headers=headers,
            json=datas,
        )
        toc = time.time()
        print(f"Time: {toc - tic:.3f}s")

        print("detection result:")
        print(response.json())

        # visualize
        res = response.json()
        print(f"response: {res['text']}")
        if "edited_image" in res:
            image_base64 = res["edited_image"]
            image = base64_to_pil(image_base64)
            image.save("/mnt/petrelfs/share_data/suzhaochen/new_tool/Tool-Factory/tool_server/tool_workers/online_workers/test_cases/truck_segmented_test.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # worker parameters
    parser.add_argument(
        "--controller-address", type=str, default="http://SH-IDCA1404-10-140-54-2:20001"
    )
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default='SegmentRegionAroundPoint')

    # model parameters
    parser.add_argument(
        "--point_response", type=str, default="<point x=\"63.0\" y=\"46.5\" alt=\"truck in the scene\">truck in the scene</point>"
    )
    parser.add_argument(
        "--send_image", action="store_true",
    )
    parser.add_argument(
        "--image_path", type=str, default="/mnt/petrelfs/songmingyang/code/reasoning/tool-agent/tool_server/tool_workers/online_workers/test_cases/mathvista_35.jpg"
    )

    # parallel test parameters
    parser.add_argument(
        "--parallel", action="store_true", help="Run parallel testing"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="Number of parallel requests"
    )
    
    args = parser.parse_args()
    args.send_image = True  # Always send image for this test
    args.parallel = True  # Always run in parallel for this test
    
    main()
