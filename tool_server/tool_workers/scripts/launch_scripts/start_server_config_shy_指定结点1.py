import os
import time
import subprocess
import logging
from pathlib import Path
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict
from box import Box
import argparse
import yaml

from tool_server.utils.utils import load_json_file, write_json_file

class ServerManager:
    """服务器管理类，用于管理和协调多个服务进程"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = Box(config)
        self.logger = self._setup_logger()
        self.log_folder = Path(self.config.log_folder)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        self.controller_addr = None
        self._clean_environment()
        os.chdir(self.config.base_dir)
        
        self.controller_config = self.config.controller_config
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []
        self.slurm_job_ids = []

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _clean_environment(self) -> None:
        os.environ.pop('SLURM_JOB_ID', None)
        os.environ["OMP_NUM_THREADS"] = "1"

    def run_srun_command(self, job_name: str, gpus: int, cpus: int, 
                         command: List[str], log_file: str, srun_kwargs: Dict = {}, 
                         conda_env: str = None, cuda_visible_devices: Optional[str] = None, 
                         node_list: Optional[str] = None, use_apptainer: bool = False, 
                         apptainer_image: Optional[str] = None) -> subprocess.Popen: # node_list 允许为 None
        """
        通过SLURM运行命令
        
        参数:
            job_name: 作业名称
            gpus: 需要的GPU数量
            cpus: 需要的CPU数量
            command: 要执行的命令列表
            log_file: 日志文件路径
            srun_kwargs: 额外的srun参数
            conda_env: Conda环境名称，如果需要激活特定环境
            cuda_visible_devices: 控制CUDA可见设备，如果为None则不设置，如果为空字符串则清除
            node_list: 指定运行的节点列表，如果为None则不指定
            use_apptainer: 是否使用apptainer容器
            apptainer_image: apptainer镜像路径
            
        返回:
            subprocess.Popen对象，表示运行的进程
        """
        srun_cmd = [
            "srun",
            f"--partition={self.config.partition}",
            f"--job-name={job_name}",
            "--mpi=pmi2",
            # f"--gres=gpu:{gpus}",
            "--gres=gpu:0",
            "-n1",
            "--ntasks-per-node=1",
            # f"-c {cpus}",
            # 手动指定cpu数量
            "-c 16",
            "--kill-on-bad-exit=1",
            "--quotatype=reserved",
            # "--quotatype=spot",
            f"--output={log_file}",
        ]
        
        # 仅当 node_list 不为 None 时才添加 --nodelist 参数
        if node_list is not None: 
            srun_cmd.append(f"--nodelist={node_list}")
            
        if conda_env:
            srun_cmd.insert(0, f"source ~/miniconda3/bin/activate {conda_env} &&")
            
        if cuda_visible_devices is not None and cuda_visible_devices != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            self.logger.info(f"设置 CUDA_VISIBLE_DEVICES 为: {os.environ['CUDA_VISIBLE_DEVICES']} for job {job_name}")
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
            self.logger.info(f"清除 CUDA_VISIBLE_DEVICES 环境变量 for job {job_name}")

        for k,v in srun_kwargs.items():
            srun_cmd.extend([f"-{k}", str(v)])
        
        # 如果使用apptainer，则包装命令
        if use_apptainer and apptainer_image:
            # 设置apptainer需要的环境变量
            apptainer_env = os.environ.copy()
            apptainer_env.update({
                "NCCL_SOCKET_IFNAME": "bond0",
                "NCCL_IB_HCA": "mlx5_0", 
                "NCCL_DEBUG": "ERROR",
                "NCCL_DEBUG_SUBSYS": "ALL",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "OMP_NUM_THREADS": "8",
                "CFLAGS": "-I/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include",
                "LDFLAGS": "-L/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/lib",
                "C_INCLUDE_PATH": "/mnt/petrelfs/sunhaoyu/visual-code/libaio/usr/include"
            })
            
            apptainer_cmd = [
                "apptainer", "exec",
                "--nv",
                "--bind", "/mnt:/mnt",
                apptainer_image
            ]
            
            srun_cmd.extend(apptainer_cmd)
            srun_cmd.extend(command)
            
            self.logger.info(f"启动作业: {job_name} 使用apptainer镜像 {apptainer_image}")
            self.logger.info(f"完整的srun+apptainer命令: {' '.join(srun_cmd)}")
            
            return subprocess.Popen(" ".join(srun_cmd), shell=True, env=apptainer_env)
        else:
            srun_cmd.extend(command)
            
            self.logger.info(f"启动作业: {job_name} 使用conda环境 {conda_env if conda_env else '原始环境'}")
            self.logger.info(f"完整的srun命令: {' '.join(srun_cmd)}")
            
            return subprocess.Popen(" ".join(srun_cmd), shell=True, env=os.environ.copy())

    def wait_for_job(self, job_name: str) -> str:
        """
        等待作业分配资源并获取节点信息
        """
        self.logger.info(f"等待作业启动: {job_name}")
        while True:
            job_id = subprocess.getoutput(
                f"squeue --user=$USER --name={job_name} --noheader --format='%A' | head -n 1"
            ).strip()
            
            if job_id:
                node_list = subprocess.getoutput(
                    f"squeue --job={job_id} --noheader --format='%N'"
                ).strip()
                if node_list:
                    self.logger.info(f"作业 {job_name} 正在节点上运行: {node_list}")
                    return {"job_id": job_id, "node_list": node_list}
            
            time.sleep(self.config.retry_interval)

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """
        等待工作器地址可用
        """
        self.logger.info(f"等待 {worker_name} 工作器准备就绪...")
        attempt = 0
        
        while True:
            try:
                attempt += 1
                response = requests.post(
                    f"{self.controller_addr}/get_worker_address",
                    json={"model": worker_name},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                
                address = response.json().get("address", "")
                if address.strip():
                    self.logger.info(f"工作器 {worker_name} 已就绪，地址: {address}")
                    return address
                
                self.logger.warning(f"尝试 {attempt}: 工作器尚未就绪")
                
            except Exception as e:
                self.logger.error(f"尝试 {attempt} 失败: {e}")
            
            time.sleep(self.config.retry_interval)

    def start_controller(self) -> str:
        """
        启动控制器服务
        """
        log_file = self.log_folder / f"{self.controller_config.worker_name}.log"
        script_addr = self.controller_config.cmd.pop("script-addr")
        job_name = self.controller_config.job_name
        
        command = ["python", script_addr]
        for k, v in self.controller_config.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        # 控制器默认在指定节点上运行，除非配置文件中明确设置为 None
        controller_node = self.controller_config.get("node_list", "SH-IDC1-10-140-37-138")
        controller_cuda_devices = self.controller_config.get("cuda_visible_devices", None) 

        self.run_srun_command(
            job_name, 
            gpus=0, 
            cpus=self.config.default_control_cpus, 
            command=command, 
            log_file=str(log_file), 
            srun_kwargs=self.controller_config.get("srun_kwargs", {}), 
            conda_env=self.controller_config.get("conda_env", None), 
            cuda_visible_devices=controller_cuda_devices,
            node_list=controller_node # 控制器可以指定节点
        )
        
        wait_dict = self.wait_for_job(job_name)
        
        node_list = wait_dict["node_list"]
        job_id = wait_dict["job_id"]
        
        self.slurm_job_ids.append(job_id)
        
        self.controller_addr = f"http://{node_list}:{self.controller_config.cmd.port}"
        self.logger.info(f"控制器正在运行，地址: {self.controller_addr}")
        
        controller_addr_dict = {"controller_addr": self.controller_addr}
        if "controller_addr_location" in self.controller_config:
            self.controller_addr_location = self.controller_config.controller_addr_location
            write_json_file(controller_addr_dict, self.controller_config.controller_addr_location)
            self.logger.info(f"控制器地址已保存到: {self.controller_addr_location}")
        else:
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../../online_workers/controller_addr/controller_addr.json"
            write_json_file(controller_addr_dict, self.controller_addr_location)
            self.logger.info(f"控制器地址已保存到: {self.controller_addr_location}")
            
        return self.controller_addr

    def start_all_workers(self) -> None:
        """启动所有工作器服务"""
        self.start_model_worker()
        self.start_tool_worker()

    def start_model_worker(self) -> None:
        """启动所有模型工作器"""
        for config_item in self.model_worker_config:
            config_obj = list(config_item.values())[0]
            self.start_worker_by_config(config_obj)
    
    def start_tool_worker(self) -> None:
        """启动所有工具工作器"""
        for config_item in self.tool_worker_config:
            config_obj = list(config_item.values())[0]
            self.start_worker_by_config(config_obj)
    
    def start_worker_by_config(self, config_obj: Box) -> None:
        """
        根据配置启动特定工作器
        """
        if "dependency_worker_name" in config_obj:
            self.wait_for_job(config_obj.dependency_worker_name)
            
        log_file = self.log_folder / f"{config_obj.worker_name}_worker.log"
        script_addr = config_obj.cmd.pop("script-addr")
        job_name = config_obj.job_name
        
        command = [
            "python", "-m", script_addr,
            "--controller_addr", self.controller_addr,
            "--model_name", config_obj.worker_name,
        ]
        
        for k, v in config_obj.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        gpus = 0 
        cpus = 0 
        cuda_visible_devices = config_obj.get("cuda_visible_devices", None) 
        
        worker_node = None # 默认不指定节点
        
        if config_obj.calculate_type == "control":
            self.logger.info(f"启动控制型工作器 {job_name}。系统自动分配节点，不使用GPU。")
            cpus = self.config.default_control_cpus
            # worker_node 保持为 None
            
        elif config_obj.calculate_type == "calculate":
            self.logger.info(f"启动计算型工作器 {job_name}。指定节点为 {config_obj.get('node_list', 'SH-IDC1-10-140-37-138')}，使用 CUDA_VISIBLE_DEVICES。")
            cpus = self.config.default_calculate_cpus
            # 计算型任务从配置中获取或使用默认指定节点
            worker_node = config_obj.get("node_list", "SH-IDC1-10-140-37-138") 
        else:
            raise ValueError("计算类型必须是 'control' 或 'calculate'")
        
        self.run_srun_command(
            job_name, 
            gpus, 
            cpus, 
            command, 
            str(log_file), 
            srun_kwargs=config_obj.get("srun_kwargs", {}), 
            conda_env=config_obj.get("conda_env", None), 
            cuda_visible_devices=cuda_visible_devices, 
            node_list=worker_node, # 根据 calculate_type 传入可能为 None 的 worker_node
            use_apptainer=config_obj.get("use_apptainer", False),
            apptainer_image=config_obj.get("apptainer_image", None)
        )
        
        wait_dict = self.wait_for_job(job_name)
        job_id = wait_dict["job_id"]
        
        self.slurm_job_ids.append(job_id)
        
        if "wait_for_self" in config_obj and config_obj["wait_for_self"]:
            self.wait_for_worker_addr(config_obj.worker_name)

    def shutdown_services(self) -> None:
        """
        关闭所有SLURM服务
        """
        if not hasattr(self, 'slurm_job_ids') or not self.slurm_job_ids:
            self.logger.warning("未找到需要关闭的SLURM作业ID")
            return
            
        try:
            os.remove(self.controller_addr_location)
            self.logger.info("控制器地址文件已删除")
        except FileNotFoundError:
            self.logger.warning("未找到控制器地址文件，跳过删除")
        except Exception as e:
            self.logger.warning(f"删除控制器地址文件时发生错误: {e}")
            
        try:
            for job_id in self.slurm_job_ids:
                try:
                    check_cmd = f"squeue --job={job_id} --noheader"
                    if subprocess.getoutput(check_cmd).strip():
                        subprocess.run(["scancel", str(job_id)], check=True)
                        self.logger.info(f"成功取消作业ID: {job_id}")
                    else:
                        self.logger.info(f"作业ID: {job_id} 已经结束")
                        
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"取消作业ID {job_id} 出错: {e}")
                except Exception as e:
                    self.logger.error(f"取消作业ID {job_id} 时发生意外错误: {e}")
                    
            self.slurm_job_ids.clear()
            self.logger.info("所有服务已经关闭")
            
        except Exception as e:
            self.logger.error(f"关闭过程中发生严重错误: {e}")
            raise
        finally:
            if 'SLURM_JOB_ID' in os.environ:
                del os.environ['SLURM_JOB_ID']


def main():
    """主函数，脚本入口点"""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml", help="配置文件路径")
    
    args = argparser.parse_args()
    config_path = Path(args.config)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        manager = ServerManager(config)
        
        os.chdir(manager.config.base_dir)
        
        manager.start_controller()
        manager.start_all_workers()
        
        try:
            while True:
                time.sleep(1)
            
        except KeyboardInterrupt:
            logger = logging.getLogger(__name__)
            logger.info("正在关闭服务...")
            manager.shutdown_services()
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"发生错误: {e}")
        raise

if __name__ == "__main__":
    main()