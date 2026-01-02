import random
import requests
import numpy as np
from time import sleep
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

class RAGEngineEnv(ToolBase):
    name = "rag"

    url = "http://10.39.9.15:8000/retrieve"
    topk = 3
    action_start = '<search>'
    action_end = '</search>'
    answer_start = '<answer>'
    answer_end = '</answer>'
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return '', 0.0, True, {}

        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_list:
            # print(f' [DEBUG] no action_list in {action_string=}')
            return '',  0.0, True, {}

        search_results = self._batch_search(action_list)
        if 'result' not in search_results or not search_results['result']:
            print(f' [WARNING] {action_list=} has no search result : {search_results}')
            return 'SEARCH RESULT IS EMPTY', 0.0, False, search_results

        assert len(action_list) == len(search_results['result']), f'{action_list=}, {len(search_results["result"])=}'
        doc_string = self._passages2string(action_list, search_results['result'])
        docs_string = f"\n\n<information>{doc_string}</information>\n\n"
        return docs_string, 0.0, False, search_results

    def reset(self, *args, **kwargs):
        pass

    def _batch_search(self, queries, max_retry=32):
        payload = {
            "queries": queries,
            "topk": self.topk,
            "return_scores": True
        }
        
        for it in range(max_retry):
            try:
                resp = requests.post(self.url, json=payload)
                resjson = resp.json()
                return resjson
            except Exception as err:
                print(f' [ERROR] err={err} -- retry for {it}')
                sleep(random.uniform(0.1, 1.0))
                continue
        return {}

    def _passages2string(self, search_keys, retrieval_result):
        format_reference = ''
        for key, results in zip(search_keys, retrieval_result):
            for idx, doc_item in enumerate(results):
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
                # format_reference += f"Doc {idx+1}\nKeyword: {key}\nTitle: {title}\nContent: {text}"
                format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
