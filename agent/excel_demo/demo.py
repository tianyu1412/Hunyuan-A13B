# -*- coding: utf-8 -*-
import io
import os
import re
import sys
import json
import uuid
import base64
import random
import string
import nbformat
import nbclient
from PIL import Image
import traceback
from openai import OpenAI


class RequestHandle:
    def __init__(self, mdoel_name, url):
        self.model_name = mdoel_name
        self.url = url

    def request(self, messages):
        openai_api_base = f"{self.url}/v1"
        openai_api_key = "EMPTY"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "代码解释器，执行代码，返回结果",
                "parameters": {
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "代码块"
                        }
                    },
                    "required": ["code"],
                    "type": "object"
                }
            }
        }]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=0.5,
            top_p=0.7,
            tools=tools,
            max_tokens=8192,
            extra_body={
                "repetition_penalty": 1.05,
                "top_k": 20
            },
        )
        response = response.model_dump()
        text = response["choices"][0]["message"]["content"]
        if "</think>" in text:
            think = text[text.find("<think>") + len("<think>"): text.rfind("</think>")].strip()
            response["choices"][0]["message"]["reasoning_content"] = think
        resp = None
        if "</answer>" in text:
            resp = text[text.find("<answer>") + len("<answer>"): text.rfind("</answer>")].strip()
            if resp.startswith("助手："):
                resp = resp[len("助手："):].strip()
        response["choices"][0]["message"]["content"] = resp
        return response


class TestClient(object):
    def __init__(self, request_handle):
        super().__init__()
        self.request_handle = request_handle
        self.max_iter = 15
        self.nb = nbformat.v4.new_notebook()
        self.nb_client = nbclient.NotebookClient(self.nb, timeout=600)
        if self.nb_client.kc is None or not self.nb_client.kc.is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    def get_system_prompt(self):
        new_system_prompt = "对于所有文件而言，默认的所在目录是./excels/ 。你可以通过文件的url来读取文件。"
        return new_system_prompt

    def remove_escape_and_color_codes(self, input_str: str):
        pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
        result = pattern.sub("", input_str)
        return result

    def parse_nbclient_output(self, outputs):
        res_list = []
        for i, output in enumerate(outputs):
            data = {}
            if output["output_type"] == "stream" and output["name"] == "stderr":
                data["type"] = "stderr"
                data["content"] = output["text"]
            elif output["output_type"] == "display_data" and "image/png" in output["data"]:
                data["type"] = "image"
                data["content"] = output["data"]["image/png"]
            elif output["output_type"] == "execute_result":
                data["type"] = "text"
                data["content"] = self.remove_escape_and_color_codes(output["data"]["text/plain"])
            elif output["output_type"] == "error":
                error_text = "\n".join(output["traceback"])
                data["type"] = "error"
                data["content"] = self.remove_escape_and_color_codes(error_text)
            else:
                continue
            res_list.append(data)
        for r in res_list:
            if r["type"] == "error":
                res_list = [r]
                break
        return res_list

    def _code_preprocess(self, code, other_info, file_path):
        if "url" in other_info:
            code = code.replace(other_info["url"], file_path)
        if "file_path" in other_info:
            code = code.replace(other_info["file_path"], file_path)
        code = code.replace(".head()", "")
        code = code.replace("SimHei", "Source Han Sans CN")

        if "import pandas as pd" in code:
            code = code.replace("import pandas as pd",
                                'import pandas as pd\npd.set_option("display.unicode.ambiguous_as_wide", '
                                'True)\npd.set_option("display.unicode.east_asian_width", True)\npd.set_option('
                                '"display.min_rows", 20)\npd.set_option("display.max_rows", 20)')

        code_list = []
        for c in code.split('\n'):
            if "locals()" in code:
                var = c.strip()
                var = var.lstrip("(")
                var = var.rstrip(")")
                if c.startswith(
                        "    ") and "#" not in var and ":" not in var and "=" not in var and "import" not in var \
                        and var != "" and "(" not in var and ")" not in var \
                        and var not in ["break", "continue", "pass", "return", "yield", "assert"]:
                    c = c.replace(var, f"print({var})")
            code_list.append(c)
        code = "\n".join(code_list)
        return code

    def _code_interpreter(self, code):
        cell = nbformat.v4.new_code_cell(source=code)
        self.nb.cells.append(cell)
        try:
            self.nb_client.execute_cell(cell, len(self.nb.cells) - 1)
        except:
            pass
        res_list = self.parse_nbclient_output(self.nb.cells[-1].outputs)
        return res_list

    def _parse_result(self, result, other_info, file_path):
        code = None
        result_list = None
        if result.get("tool_calls", None) is not None and len(result["tool_calls"]) > 0:
            action = result["tool_calls"]
            code = self._code_preprocess(
                json.loads(action[0]["function"]["arguments"])["code"],
                other_info, file_path
            )
            res_list = self._code_interpreter(code)
            result_list = []
            for res in res_list:
                if res["type"] in ["text", "error"]:
                    result_list.append(res["content"])
                elif res["type"] == "image":
                    result_list.append("[IMAGE]")
                    self._process_image(res["content"], ".")
            result_list = "\n".join(result_list)
            result = "%s" % (
                json.dumps({"code_result": result_list}, ensure_ascii=False)
            )
        return result, code, result_list

    def _process_image(self, image_str, thread_id_path="."):
        encoded_data = image_str.encode("utf-8")  # str -> base64
        decoded_data = base64.b64decode(encoded_data)  # base64 -> bin
        image_bytes = io.BytesIO(decoded_data)
        image = Image.open(image_bytes)
        image_name = "image-{}.jpg".format(
            "".join(random.choice(string.ascii_letters + string.digits) for _ in range(24)))
        image_path = os.path.join(thread_id_path, image_name)
        image.convert("RGB").save(image_path)
        image_markdown = f"![Picture]({image_path})"
        return image_markdown

    def message_combine(self, file_name, question):
        other_info = {
            "file_name": file_name
        }
        url = "https://hunyuan.tencent.com/files/" + str(uuid.uuid4())
        other_info.update({"url": url})
        other_info.update({"file_path": f"./excels/{file_name}"})
        message = f"文件类型：Excel\n文件名：{file_name}\n文件URL地址：{url}\n" + question
        return message, other_info

    def __call__(self, input_data):
        output_data = {}
        try:
            messages = [{
                "role": "system",
                "content": self.get_system_prompt()
            }]
            reasons = []
            other_info = {}
            while True:
                question = input("(请输入你的问题，输入exit退出) 用户：")
                if question.strip() == "exit":
                    break
                message, other_info = self.message_combine(input_data["file_name"], question)
                print("---" * 10)
                messages.append({"role": "user", "content": message})
                for _ in range(self.max_iter):
                    resp = self.request_handle.request(messages)
                    resp = resp["choices"][0]["message"]
                    reason = resp["reasoning_content"] if "reasoning_content" in resp else None
                    if reason is not None:
                        print("推理过程:\n" + reason.strip())
                        reasons.append(reason)
                    messages.append({"role": "assistant", "content": resp.get("content", None),
                                     "tool_calls": resp.get("tool_calls", None)})
                    message, code, result_list = self._parse_result(resp, other_info, input_data["file_path"])
                    if code is None:
                        print("---" * 3)
                        print("助手：", message["content"])
                        break
                    else:
                        print("---" * 3 + "\n执行代码：\n" + code)
                        print("---" * 10)
                        assert result_list is not None
                        print("执行结果：\n" + result_list)
                        messages.append({"role": "user", "content": message})
                    countinue_flag = input("继续执行吗？(输入exit退出)")
                    if countinue_flag.strip() == "exit":
                        break
        except:
            e = traceback.format_exc()
            print(e)
            output_data["error_message"] = e
            return output_data
        output_data["messages"] = messages
        output_data["reasons"] = reasons
        return output_data


if __name__ == "__main__":
    handle = RequestHandle(sys.argv[1], sys.argv[2])
    file_path = sys.argv[3] if sys.argv[3] else input("输入 xlsx/xls 文件路径：")
    assert file_path.endswith(".xlsx") or file_path.endswith(".xls"), "文件格式错误"
    assert os.path.exists(file_path), "excel 文件不存在"
    input_data = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path
    }
    api = TestClient(handle)
    output_data = api(input_data)
