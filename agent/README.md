HunYuan-A13B Agent 功能介绍

## 📖 简介
HunYuan-A13B 模型支持通过函数调用（Function Call）来实现 Agent 的搭建。本文档主要介绍 HunYuan-A13B 函数调用能力的使用方法和原理介绍，并给出两个 Agent 搭建示例，包括：MCP Agent 和 Excel Agent。

## 🚀 快速开始

### 函数定义

#### 函数整体结构
```json
{
    "type": "function",
    "function": {
        "name": "函数名称",
        "description": "函数功能描述",
        "parameters": {
            "type": "object",
            "properties": {
                [简单类型参数]
                [简单枚举类型参数]
                [数组类型参数]
                [复杂对象类型参数]
            }
            "required": []
        }
    }
}
```

函数共三个字段：type、function。
- `type`: 固定为 function
- `function`: 描述了具体的函数，包括三个字段：name、description、parameters
    - `name`: 函数名称
    - `description`: 函数功能描述
    - `parameters`: 描述了函数的输入参数信息，包括三个字段：type、properties、required
      - `type`: 参数类型，固定为 object 类型
      - `properties`: 参数属性，可以是[简单类型参数]、[简单枚举类型参数]、[数组类型参数]、[复杂对象类型参数]的任意组合，数量不限
      - `required`: 必填参数，填写 properties 里的必填参数

#### 简单类型参数结构

```json
"parameter_name": {
    "type": "string|integer|float|boolean",
    "description": "参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。"
}
```

简单类型参数共二个字段：type、description
- `type`: 参数类型，包括：string、integer、float、boolean这四种类型。
- `description`: 参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。

#### 简单枚举类型参数结构

```json
"parameter_name": {
    "type": "string|integer|float",
    "description": "参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。",
    "enum": []
}
```
简单枚举类型参数共三个字段：type、description、enum
- `type`: 参数类型，包括：string、integer、float这三种类型。
- `description`: 参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。
- `enum`: 参数枚举值，参数值只能填写该列表中的任意一个结果。

#### 数组类型参数结构

```json
"parameter_name": {
    "type": "array",
    "description": "参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。",
    "items": {
        [简单类型参数]
        [简单枚举类型参数]
        [数组类型参数]
        [复杂对象类型参数]
    }
}
```

数组类型参数共三个字段：type、description、items
- `type`: 参数类型，固定为 array 类型。
- `description`: 参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。
- `items`: 数组元素，可以是[简单类型参数]、[数组类型参数]、[简单枚举类型参数]、[复杂对象类型参数]的任意一种。

#### 复杂对象类型参数结构

```json
"parameter_name": {
    "type": "object",
    "description": "参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。",
    "properties": {
        [简单类型参数]
        [简单枚举类型参数]
        [数组类型参数]
        [复杂对象类型参数]
    }
}
```

复杂对象类型参数共三个字段：type、description、properties
- `type`: 参数类型，固定为 object 类型。
- `description`: 参数描述，如果参数值有特殊的格式要求，请说明。此外，如果是非必填参数，则请提供默认值。
- `properties`: 复杂对象的属性，可以是[简单类型参数]、[简单枚举类型参数]、[数组类型参数]、[复杂对象类型参数]的任意组合，数量不限。

### 函数调用实现方案

我们通过设计一个特殊的 System Prompt 来实现函数调用能力，在传入 tools 字段时生效，具体可参考如下代码来查看拼接后的结果。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Ability to check the weather in any city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "description": "city name",
                        "type": "string"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    }
]

messages_with_system = [
    {"role": "system", "content": "You are a weather agent."},
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in Beijing and Shanghai?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "ShangHai"}}}
    ]},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}, {"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]'},
    {"role": "assistant", "content": "Beijing and Shanghai have sunny weather."},
    {"role": "user", "content": "Shenzhen"},
]

print("====== prompt after applying chat template ======")
print(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False))

messages_without_system = [
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in Beijing and Shanghai?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "ShangHai"}}}
    ]},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}, {"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]'},
    {"role": "assistant", "content": "Beijing and Shanghai have sunny weather."},
    {"role": "user", "content": "Shenzhen"},
]

print("====== prompt after applying chat template ======")
print(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False))
```

执行上述代码后，得到以下两个结果，该内容将作为最终输入给模型的文本请求模型。

用户传入System：
```
<|startoftext|>你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。
如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。
如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。
如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。
你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。

[{"type": "function", "function": {"name": "get_weather", "description": "Ability to check the weather in any city", "parameters": {"type": "object", "properties": {"city": {"description": "city name", "type": "string"}}, "required": ["city"]}}}]

额外要求：
You are a weather agent.

如果你决定返回函数调用，请将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>，不得包含其他文本。如果额外要求里有格式要求，请忽略，以此处为准。
否则，请参考开头说的三种情况，以助手：开头进行回复。

如果额外要求里有时间信息，就以额外要求里的时间为准，否则，参考当前时间：2025-06-26 16:21:57 星期四<|extra_4|>用户：Hi, how are you?<|extra_0|>助手：Thanks. I am doing well. How can I help you?<|eos|>用户：What's the weather like in Beijing and Shanghai?<|extra_0|><tool_calls>[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "ShangHai"}}]</tool_calls><|eos|><tool_response>[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}, {"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]</tool_response><|extra_0|>助手：Beijing and Shanghai have sunny weather.<|eos|>用户：Shenzhen<|extra_0|>
```

用户不传入System：
```
<|startoftext|>你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。
如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。
如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。
如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。
你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。

[{"type": "function", "function": {"name": "get_weather", "description": "Ability to check the weather in any city", "parameters": {"type": "object", "properties": {"city": {"description": "city name", "type": "string"}}, "required": ["city"]}}}]

如果你决定返回函数调用，请将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>，不得包含其他文本。
否则，请参考开头说的三种情况，以助手：开头进行回复。

当前时间：2025-06-26 16:22:35 星期四<|extra_4|>用户：Hi, how are you?<|extra_0|>助手：Thanks. I am doing well. How can I help you?<|eos|>用户：What's the weather like in Beijing and Shanghai?<|extra_0|><tool_calls>[{"name": "get_weather", "arguments": {"city": "Beijing"}}, {"name": "get_weather", "arguments": {"city": "ShangHai"}}]</tool_calls><|eos|><tool_response>[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}, {"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]</tool_response><|extra_0|>助手：Beijing and Shanghai have sunny weather.<|eos|>用户：Shenzhen<|extra_0|>
```

注意点：

- 根据用户是否传入 System 会使用不同的模版。

- 当开启函数调用功能（传入 tools 字段）时，chat_template 内部会自动获取当前时间，格式为%Y-%m-%d %H:%M:%S %A。

- 不建议用户在 System 里再额外增加输出格式说明，否则可能会和 chat_template 内部模版里的格式产生冲突。如果确实需要使用其他格式，请直接使用 Prompt 的方式进行调用，不传入 tools 字段，自行在 Prompt 中组织任务说明、工具信息、输出格式等内容。

模型输出结果如下：

```
<think>
...
</think>
<answer>
<tool_calls>[{"name": "get_weather", "arguments": {"city": "Shenzhen"}}]</tool_calls>
</answer>
```

函数调用结果解析的逻辑可以参考：<a href="hunyuan_tool_parser.py">hunyuan_tool_parser</a>

### Agent vLLM 推理服务部署

`vllm` 中支持 [Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling.html) 服务部署，以提供 `function call`、 `agent` 的需求。 我们 **推荐使用这种方式启动服务以更好体验模型能力**。

使用下列 `Agent 搭建示例` ，也推荐接入这种方式部署的模型服务。

启动`Tool Calling` 的 `vllm 服务` 须在启动命令中添加如下字段：

```bash
--enable-auto-tool-choice \
--tool-parser-plugin /path/to/hunyuan_tool_parser.py \
--tool-call-parser hunyuan \
```

完整启动脚本示例如下：

```bash
MODEL_PATH=${MODEL_PATH}

export VLLM_HOST_IP=$LOCAL_IP

python3 -m vllm.entrypoints.openai.api_server \
    --host ${LOCAL_IP} \
    --port 8020 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --gpu_memory_utilization 0.92 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/to/hunyuan_tool_parser.py \
    --tool-call-parser hunyuan \
    2>&1 | tee log_server.txt
```

### Agent 搭建示例

我们提供了两个使用函数调用能力来搭建 Agent 的示例，包括：MCP Agent 和 Excel Agent。具体可参考：<a href="mcp_demo/README.md">mcp_demo</a> 和 <a href="excel_demo/README.md">excel_demo</a>。

### Agent 训练

如果需要基于该模型进行微调，参考 <a href="gen_train_data.py">gen_train_data.py</a> 来构建训练数据，以下为构建得到的快思考和慢思考的训练数据示例。

快思考：
```
messages = [
    {
        "role": "system",
        "content": "你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。\n如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。\n如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。\n如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。\n你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。\n\n[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Ability to check the weather in any city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"description\": \"city name\", \"type\": \"string\"}}, \"required\": [\"city\"]}}}]\n\n额外要求：\nYou are a weather agent.\n\n如果你决定返回函数调用，请将其格式化为<tool_calls>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},...]</tool_calls>，不得包含其他文本。如果额外要求里有格式要求，请忽略，以此处为准。\n否则，请参考开头说的三种情况，以助手：开头进行回复。\n\n如果额外要求里有时间信息，就以额外要求里的时间为准，否则，参考当前时间：2025-09-16 07:34:51 星期二"
    },
    {
        "role": "user",
        "content": "/no_think用户：Hi, how are you?"
    },
    {
        "role": "assistant",
        "content": "助手：Thanks. I am doing well. How can I help you?"
    },
    {
        "role": "user",
        "content": "/no_think用户：What's the weather like in Beijing and Shanghai?"
    },
    {
        "role": "assistant",
        "content": "<think>\n\n</think>\n<answer><tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}, {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\"}}]</tool_calls></answer>"
    }
]
```

慢思考：
```
messages = [
    {
        "role": "system",
        "content": "你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。\n如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。\n如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。\n如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。\n你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。\n\n[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Ability to check the weather in any city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"description\": \"city name\", \"type\": \"string\"}}, \"required\": [\"city\"]}}}]\n\n额外要求：\nYou are a weather agent.\n\n如果你决定返回函数调用，请将其格式化为<tool_calls>[{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},...]</tool_calls>，不得包含其他文本。如果额外要求里有格式要求，请忽略，以此处为准。\n否则，请参考开头说的三种情况，以助手：开头进行回复。\n\n如果额外要求里有时间信息，就以额外要求里的时间为准，否则，参考当前时间：2025-09-16 07:34:51 星期二"
    },
    {
        "role": "user",
        "content": "用户：Hi, how are you?"
    },
    {
        "role": "assistant",
        "content": "助手：Thanks. I am doing well. How can I help you?"
    },
    {
        "role": "user",
        "content": "用户：What's the weather like in Beijing and Shanghai?"
    },
    {
        "role": "assistant",
        "content": "<think>\nOK, the user asked about the weather in Beijing and Shanghai,...\n</think>\n<answer><tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}, {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\"}}]</tool_calls></answer>"
    }
]
```

注意：
- 以上两个示例仅最后一轮 assistant 会计算 loss。
- 在不计算 loss 的历史 messages 中，我们不会拼接 `<think>\n{}\n<think>` 和 `<answer>\n{}\n<answer>`。
- 如果使用快思考模式训练，messages 中的所有 user message 都需要在开头拼接 /no_think。
