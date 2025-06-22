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
        # 初始化配置
        self.config = Box(config)  # 使用Box转换字典为对象，方便使用.访问属性
        self.logger = self._setup_logger()  # 设置日志系统
        self.log_folder = Path(self.config.log_folder)  # 创建日志文件夹路径对象
        self.log_folder.mkdir(parents=True, exist_ok=True)  # 确保日志文件夹存在
        
        # 初始化状态变量
        self.controller_addr = None  # 控制器地址，将在启动后设置
        self._clean_environment()  # 清理环境变量
        os.chdir(self.config.base_dir)  # 切换到基础目录
        
        # 从配置中提取各组件配置
        self.controller_config = self.config.controller_config  # 控制器配置
        self.model_worker_config = self.config.model_worker_config if "model_worker_config" in self.config else []  # 模型工作器配置
        self.tool_worker_config = self.config.tool_worker_config if "tool_worker_config" in self.config else []  # 工具工作器配置
        self.slurm_job_ids = []  # 存储启动的SLURM作业ID，用于后续清理

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统，返回日志器实例"""
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为INFO
            format='%(asctime)s - %(levelname)s - %(message)s'  # 设置日志格式：时间-级别-消息
        )
        return logging.getLogger(__name__)  # 返回当前模块的日志器

    def _clean_environment(self) -> None:
        """清理环境变量，避免SLURM环境变量干扰新任务"""
        os.environ.pop('SLURM_JOB_ID', None)  # 移除SLURM_JOB_ID环境变量，避免影响新作业
        os.environ["OMP_NUM_THREADS"] = "1"  # 设置OpenMP线程数为1，避免过度线程化

    def run_srun_command(self, job_name: str, gpus: int, cpus: int, 
                        command: List[str], log_file: str, srun_kwargs: Dict = {}, 
                        conda_env: str = None, cuda_visible_devices: str = None) -> subprocess.Popen:
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
            cuda_visible_devices: 控制CUDA可见设备
            
        返回:
            subprocess.Popen对象，表示运行的进程
        """
        # 构建srun命令
        srun_cmd = [
            "srun",
            f"--partition={self.config.partition}",  # 指定SLURM分区
            f"--job-name={job_name}",  # 设置作业名称
            "--mpi=pmi2",  # 设置MPI实现
            f"--gres=gpu:{gpus}",  # 请求GPU资源
            "-n1",  # 任务数量为1
            "--ntasks-per-node=1",  # 每个节点任务数为1
            f"-c {cpus}",  # 请求的CPU核心数
            "--kill-on-bad-exit=1",  # 如果任务异常退出，终止整个作业
            "--quotatype=reserved",  # 使用预留配额
            f"--output={log_file}",  # 指定输出日志文件
        ]
        
        # 如果需要激活conda环境
        if conda_env:
            srun_cmd.insert(0, f"source ~/miniconda3/bin/activate {conda_env} &&")
            
        # 如果需要设置CUDA可见设备
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        # 添加额外的srun参数
        for k,v in srun_kwargs.items():
            srun_cmd.extend([f"-{k}", str(v)])
        
        # 添加实际要运行的命令
        srun_cmd.extend(command)
        
        # 记录启动信息
        self.logger.info(f"启动作业: {job_name} 使用conda环境 {conda_env if conda_env else '原始环境'}")
        
        # 执行命令并返回进程对象
        return subprocess.Popen(" ".join(srun_cmd), shell=True, env=os.environ.copy())

    def wait_for_job(self, job_name: str) -> str:
        """
        等待作业分配资源并获取节点信息
        
        参数:
            job_name: 要等待的作业名称
            
        返回:
            包含job_id和node_list的字典
        """
        self.logger.info(f"等待作业启动: {job_name}")
        while True:
            # 获取作业ID
            job_id = subprocess.getoutput(
                f"squeue --user=$USER --name={job_name} --noheader --format='%A' | head -n 1"
            ).strip()
            
            if job_id:
                # 获取节点列表
                node_list = subprocess.getoutput(
                    f"squeue --job={job_id} --noheader --format='%N'"
                ).strip()
                if node_list:
                    self.logger.info(f"作业 {job_name} 正在节点上运行: {node_list}")
                    return {"job_id": job_id, "node_list": node_list}
            
            # 如果作业未启动，等待一段时间后重试
            time.sleep(self.config.retry_interval)

    def wait_for_worker_addr(self, worker_name: str) -> str:
        """
        等待工作器地址可用
        
        参数:
            worker_name: 工作器名称
            
        返回:
            工作器地址字符串
        """
        self.logger.info(f"等待 {worker_name} 工作器准备就绪...")
        attempt = 0
        
        while True:
            try:
                attempt += 1
                # 尝试从控制器获取工作器地址
                response = requests.post(
                    f"{self.controller_addr}/get_worker_address",
                    json={"model": worker_name},
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                
                # 解析响应，获取地址
                address = response.json().get("address", "")
                if address.strip():
                    self.logger.info(f"工作器 {worker_name} 已就绪，地址: {address}")
                    return address
                
                self.logger.warning(f"尝试 {attempt}: 工作器尚未就绪")
                
            except Exception as e:
                self.logger.error(f"尝试 {attempt} 失败: {e}")
            
            # 等待一段时间后重试
            time.sleep(self.config.retry_interval)

    def start_controller(self) -> str:
        """
        启动控制器服务
        
        返回:
            控制器服务的地址
        """
        # 设置日志文件
        log_file = self.log_folder / f"{self.controller_config.worker_name}.log"
        
        # 从配置中提取脚本地址并从命令参数中移除
        script_addr = self.controller_config.cmd.pop("script-addr")
        
        # 获取作业名称
        job_name = self.controller_config.job_name
        
        # 构建启动命令
        command = ["python", script_addr]
        for k, v in self.controller_config.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        # 执行SLURM命令启动控制器
        self.run_srun_command(
            job_name, 
            self.config.default_control_gpus, 
            self.config.default_control_cpus, 
            command, 
            str(log_file), 
            srun_kwargs=self.controller_config.get("srun_kwargs", {}), 
            conda_env=self.controller_config.get("conda_env", None), 
            cuda_visible_devices=self.controller_config.get("cuda_visible_devices", None)
        )
        
        # 等待作业启动并获取节点信息
        wait_dict = self.wait_for_job(job_name)
        
        node_list = wait_dict["node_list"]
        job_id = wait_dict["job_id"]
        
        # 记录作业ID以便后续清理
        self.slurm_job_ids.append(job_id)
        
        # 构建控制器地址
        self.controller_addr = f"http://{node_list}:{self.controller_config.cmd.port}"
        self.logger.info(f"控制器正在运行，地址: {self.controller_addr}")
        
        # 保存控制器地址到文件
        controller_addr_dict = {"controller_addr": self.controller_addr}
        if "controller_addr_location" in self.controller_config:
            # 如果配置中指定了控制器地址保存位置
            self.controller_addr_location = self.controller_config.controller_addr_location
            write_json_file(controller_addr_dict, self.controller_config.controller_addr_location)
            self.logger.info(f"控制器地址已保存到: {self.controller_addr_location}")
        else:
            # 使用默认位置
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            self.controller_addr_location = f"{current_file_path}/../../online_workers/controller_addr/controller_addr.json"
            write_json_file(controller_addr_dict, self.controller_addr_location)
            self.logger.info(f"控制器地址已保存到: {self.controller_addr_location}")
            
        return self.controller_addr

    def start_all_workers(self) -> None:
        """启动所有工作器服务"""
        self.start_model_worker()  # 启动模型工作器
        self.start_tool_worker()   # 启动工具工作器

    def start_model_worker(self) -> None:
        """启动所有模型工作器"""
        for config in self.model_worker_config:
            config = list(config.values())[0]  # 从配置字典中提取实际配置
            self.start_worker_by_config(config)  # 使用配置启动工作器
    
    def start_tool_worker(self) -> None:
        """启动所有工具工作器"""
        for config in self.tool_worker_config:
            config = list(config.values())[0]  # 从配置字典中提取实际配置
            self.start_worker_by_config(config)  # 使用配置启动工作器
    
    def start_worker_by_config(self, config) -> None:
        """
        根据配置启动特定工作器
        
        参数:
            config: 工作器配置对象
        """
        # 如果有依赖的工作器，先等待依赖的工作器启动
        if "dependency_worker_name" in config:
            self.wait_for_job(config.dependency_worker_name)
            
        # 设置日志文件
        log_file = self.log_folder / f"{config.worker_name}_worker.log"
        
        # 从配置中提取脚本地址
        script_addr = config.cmd.pop("script-addr")
        
        # 获取作业名称
        job_name = config.job_name
        
        # 构建基本命令，包括脚本路径和控制器地址
        command = [
            "python", "-m", script_addr,
            "--controller_addr", self.controller_addr,
            "--model_name", config.worker_name,
        ]
        
        # 添加其他命令行参数
        for k, v in config.cmd.items():
            command.extend([f"--{k}", str(v)])
        
        # 根据计算类型确定资源需求
        if config.calculate_type == "control":
            # 控制型工作器（一般较轻量）
            gpus = self.config.default_control_gpus
            cpus = self.config.default_control_cpus
        elif config.calculate_type == "calculate":
            # 计算型工作器（一般需要更多资源）
            gpus = self.config.default_calculate_gpus
            cpus = self.config.default_calculate_cpus
        else:
            raise ValueError("计算类型必须是 'control' 或 'calculate'")
        
        # 执行SLURM命令启动工作器
        self.run_srun_command(
            job_name, 
            gpus, 
            cpus, 
            command, 
            str(log_file), 
            srun_kwargs=config.get("srun_kwargs", {}), 
            conda_env=config.get("conda_env", None), 
            cuda_visible_devices=config.get("cuda_visible_devices", None)
        )
        
        # 等待作业启动并获取作业ID
        wait_dict = self.wait_for_job(job_name)
        job_id = wait_dict["job_id"]
        
        # 记录作业ID以便后续清理
        self.slurm_job_ids.append(job_id)
        
        # 如果需要，等待工作器自身准备就绪
        if "wait_for_self" in config and config["wait_for_self"]:
            self.wait_for_worker_addr(config.worker_name)

    def shutdown_services(self) -> None:
        """
        关闭所有SLURM服务
        
        使用记录的作业ID列表逐一关闭服务，处理错误并记录日志
        """
        # 检查是否有作业ID需要清理
        if not hasattr(self, 'slurm_job_ids') or not self.slurm_job_ids:
            self.logger.warning("未找到需要关闭的SLURM作业ID")
            return
            
        # 尝试删除控制器地址文件
        try:
            os.remove(self.controller_addr_location)
            self.logger.info("控制器地址文件已删除")
        except:
            self.logger.warning("未找到控制器地址文件，跳过删除")
            pass
            
        try:
            # 逐一取消作业
            for job_id in self.slurm_job_ids:
                try:
                    # 检查作业是否仍在运行
                    check_cmd = f"squeue --job={job_id} --noheader"
                    if subprocess.getoutput(check_cmd).strip():
                        # 取消作业
                        subprocess.run(["scancel", str(job_id)], check=True)
                        self.logger.info(f"成功取消作业ID: {job_id}")
                    else:
                        self.logger.info(f"作业ID: {job_id} 已经结束")
                        
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"取消作业ID {job_id} 出错: {e}")
                except Exception as e:
                    self.logger.error(f"取消作业ID {job_id} 时发生意外错误: {e}")
                    
            # 清空作业ID列表
            self.slurm_job_ids.clear()
            self.logger.info("所有服务已经关闭")
            
        except Exception as e:
            self.logger.error(f"关闭过程中发生严重错误: {e}")
            raise
        finally:
            # 确保环境变量被清理
            if 'SLURM_JOB_ID' in os.environ:
                del os.environ['SLURM_JOB_ID']


def main():
    """主函数，脚本入口点"""
    # 解析命令行参数
    argparser = argparse.ArgumentParser()
    # 设置默认配置文件路径
    argparser.add_argument("--config", type=str, default="/mnt/petrelfs/sunhaoyu/visual-code/Tool-Factory-Filter/tool_server/tool_workers/scripts/launch_scripts/config/all_service_example.yaml", help="配置文件路径")
    
    args = argparser.parse_args()
    config_path = Path(args.config)
    
    # 加载配置文件
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    try:
        # 创建服务管理器
        manager = ServerManager(config)
        
        # 切换到基础目录
        os.chdir(manager.config.base_dir)
        
        # 启动控制器
        manager.start_controller()
        
        # 启动所有工作器
        manager.start_all_workers()
        
        try:
            # 保持运行，直到被中断
            while True:
                time.sleep(1)
            
        except KeyboardInterrupt:
            # 捕获键盘中断信号，优雅地关闭服务
            logger = logging.getLogger(__name__)
            logger.info("正在关闭服务...")
            manager.shutdown_services()
            
    except Exception as e:
        # 捕获所有其他异常
        logger = logging.getLogger(__name__)
        logger.error(f"发生错误: {e}")
        raise

if __name__ == "__main__":
    main()
