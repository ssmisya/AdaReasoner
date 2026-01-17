source ~/.bashrc
source ~/anaconda3/bin/activate toolrl

node=your_node

unset http_proxy
unset HTTP_PROXY


job_id=$1

srun -p ai_moe -w ${node} ray job stop ${job_id} --address=http://${node}:10021

