from huggingface_hub import upload_folder
from huggingface_hub import HfApi
from tool_server.utils.utils import setup_openai_proxy,setup_proxy
api = HfApi()
setup_proxy()
# upload_folder(
#     folder_path="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_dataset_new",
#     repo_id="hitsmy/tool_pathverify_v12",
#     repo_type="dataset",
#     token="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",  # 可选，如果已登录可以不填
#     # repo_exists_ok=True ,
# )

api.upload_large_folder(
    repo_id="hitsmy/tool_pathverify_v112",
    repo_type="dataset",
    folder_path="/mnt/petrelfs/songmingyang/code/reasoning/tool_util_code/frozen_lake/data_curation/frozen_lake_dataset_new",
)