import json
import random
import time

from datetime import datetime


system_prompt_template_with_extra_system = '''
你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。
如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。
如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。
如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。
你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。

{{{tools}}}

额外要求：
{{{extra_system_prompt}}}

如果你决定返回函数调用，请将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>，不得包含其他文本。如果额外要求里有格式要求，请忽略，以此处为准。
否则，请参考开头说的三种情况，以助手：开头进行回复。

如果额外要求里有时间信息，就以额外要求里的时间为准，否则，参考当前时间：{{{env_info}}}
'''.strip("\n")


system_prompt_template_without_extra_system = '''
你是一位函数组合专家。你会得到一个问题和一组可能的函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的。
如果没有一个函数可以使用，请直接使用自然语言回复用户，以助手：开头。
如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息，以助手：开头。
如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户，以助手：开头。
你应该只在工具调用部分返回函数调用。如果你决定调用任何函数，你必须将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>。你不应该在回复中包含任何其他文本。以下是你可以调用的函数列表，格式为JSON。

{{{tools}}}

如果你决定返回函数调用，请将其格式化为<tool_calls>[{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},...]</tool_calls>，不得包含其他文本。
否则，请参考开头说的三种情况，以助手：开头进行回复。

当前时间：{{{env_info}}}
'''.strip("\n")


def get_random_date(language="zh"):
    a1 = (2024, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（1976-01-01 00：00：00）
    a2 = (2025, 12, 31, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组（1990-12-31 23：59：59）

    start = time.mktime(a1)  # 生成开始时间戳
    end = time.mktime(a2)  # 生成结束时间戳

    t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(t)  # 将时间戳生成时间元组
    date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)  # 将时间元组转成格式化字符串（1976-05-21）
    date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    weekday_num = date_obj.weekday()
    if language == "zh":
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    else:
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekdays[weekday_num]
    date = date + " " + weekday
    return date


def process_messages(env_info, tools, messages):
    messages_new = []
    functions_str = json.dumps(tools, ensure_ascii=False)
    if messages[0]["role"] == "system":
        extra_system_prompt = messages[0]["content"]
        system_prompt = system_prompt_template_with_extra_system.replace("{{{tools}}}", functions_str) \
                                                                .replace("{{{env_info}}}", env_info) \
                                                                .replace("{{{extra_system_prompt}}}", extra_system_prompt)
    else:
        system_prompt = system_prompt_template_without_extra_system.replace("{{{tools}}}", functions_str) \
                                                                   .replace("{{{env_info}}}", env_info)
    messages_new.append({"role": "system", "content": system_prompt})

    last_role = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        reasoning_content = message.get("reasoning_content", "")
        if role == "user":
            content_new = f"用户：{content}"
            message_new = {"role": "user", "content": content_new}
            messages_new.append(message_new)
        elif role == "assistant":
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                action_list = []
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    function_name = function["name"]
                    function_arguments = function["arguments"]
                    if isinstance(function_arguments, str):
                        function_arguments = json.loads(function_arguments)
                    action_list.append({"name": function_name, "arguments": function_arguments})
                action_list = json.dumps(action_list, ensure_ascii=False)
                content_new = f"<tool_calls>{action_list}</tool_calls>"
            else:
                content_new = f"助手：{content}"
            message_new = {"role": "assistant", "reasoning_content": reasoning_content, "content": content_new}
            messages_new.append(message_new)
        elif role == "tool":
            if last_role == "tool":
                # 处理连续多个tool的情况
                last_tool_observation = messages_new[-1]["content"].replace("<tool_response>", "").replace(
                    "</tool_response>", "")
                last_tool_observation = json.loads(last_tool_observation)
                last_tool_observation.append(json.loads(content))
                content = json.dumps(last_tool_observation, ensure_ascii=False)
                content_new = f"<tool_response>{content}</tool_response>"
                messages_new[-1]["content"] = content_new
            else:
                content_new = f"<tool_response>{content}</tool_response>"
                message_new = {"role": "user", "content": content_new}
                messages_new.append(message_new)

        last_role = role
    return messages_new


def split_multi_turn(messages):
    messages_history = []
    messages_split = []
    system_message = messages[0]
    current_messages = [system_message]
    for message in messages[1:]:
        role = message["role"]
        content = message["content"]
        reasoning_content = message.get("reasoning_content", "")
        if role == "user":
            content = f"/no_think{content}"
            current_messages.append({"role": "user", "content": content})
            messages_history.append({"role": "user", "content": content})

        elif role == "assistant":
            content_new = f"<think>\n{reasoning_content}\n</think>\n<answer>{content}</answer>"
            current_messages.append({"role": "assistant", "content": content_new})
            messages_history.append({"role": "assistant", "content": content})
            messages_split.append(current_messages)
            current_messages = [system_message] + messages_history
    return messages_split


env_info = get_random_date()

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

# quick think
messages_with_system_quick_think = [
    {"role": "system", "content": "You are a weather agent."},
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in Beijing and Shanghai?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Shanghai"}}}
    ]},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}]'},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]'},
    {"role": "assistant", "content": "Beijing and Shanghai have sunny weather."}
]

messages_new = process_messages(env_info, tools, messages_with_system_quick_think)
messages_split = split_multi_turn(messages_new)
print(json.dumps(messages_split, ensure_ascii=False, indent=4))

# slow think
messages_with_system_slow_think = [
    {"role": "system", "content": "You are a weather agent."},
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "reasoning_content": "OK, the user sent 'Hi, how are you?', ...", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in Beijing and Shanghai?"},
    {"role": "assistant", "reasoning_content": "OK, the user asked about the weather in Beijing and Shanghai,...", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Shanghai"}}}
    ]},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Beijing", "weather": "sunny"}}]'},
    {"role": "tool", "content": '[{"status_code": 200, "weather_info": {"city": "Shanghai", "weather": "sunny"}}]'},
    {"role": "assistant", "reasoning_content": "OK, the user asked about the weather in Beijing and Shanghai, and the weather in Beijing and Shanghai has been found to be sunny, ...", "content": "Beijing and Shanghai have sunny weather."}
]
messages_new = process_messages(env_info, tools, messages_with_system_slow_think)
messages_split = split_multi_turn(messages_new)
print(json.dumps(messages_split, ensure_ascii=False, indent=4))
