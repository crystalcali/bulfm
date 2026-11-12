"""Native function schemas and role-based browser history for LFM3."""
import ast
import json


def function(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


BID = {"type": "string", "description": "Element bid shown in square brackets on the current page."}
TOOLS = [
    function("click", "Click an element. A readonly date input can be clicked to open its calendar.", {"bid": BID}),
    function("fill", "Replace text in an editable input. Never fill a readonly input.",
             {"bid": BID, "text": {"type": "string"}}),
    function("press", "Press a keyboard key or chord on an element.",
             {"bid": BID, "key_comb": {"type": "string"}}),
    function("select_option", "Select a value in a native select element.",
             {"bid": BID, "options": {"type": "string"}}),
    function("scroll", "Scroll the page by pixels.",
             {"delta_x": {"type": "number"}, "delta_y": {"type": "number"}}),
    function("noop", "Wait without interacting.", {}),
]
PARAMETERS = {t["function"]["name"]: list(t["function"]["parameters"]["properties"]) for t in TOOLS}


def native_system(prompt):
    # Migrate only the known legacy instruction; preserve custom task constraints.
    prompt = prompt.replace("Reply with exactly ONE Python-style action and no explanation or markdown.",
                            "Call exactly ONE provided tool for the next action. Do not return Python code.")
    start = prompt.find("Actions: click('bid')")
    end = prompt.find("The number in brackets", start)
    if start >= 0 and end >= 0:
        prompt = prompt[:start] + prompt[end:]
    return prompt.replace("=> click('13')", "=> use the click tool with bid='13'")


def observation(obs):
    return f"Last error: {obs['error']}\nCurrent page:\n{obs['tree']}"


def messages_for(prompt, obs, history):
    if not history:
        return [{"role": "system", "content": native_system(prompt)},
                {"role": "user", "content": f"Goal: {obs['goal']}\n{observation(obs)}"}]
    messages = [{"role": "system", "content": native_system(prompt)},
                {"role": "user", "content": f"Goal: {obs['goal']}\n{observation(history[0]['before'])}"}]
    for index, event in enumerate(history):
        if event['valid']:
            call = ast.parse(event['action'], mode='eval').body
            name = call.func.id
            args = dict(zip(PARAMETERS[name], [ast.literal_eval(a) for a in call.args]))
            text = event['response']
            # Preserve reasoning before the final call, without serializing the call twice.
            content = text.split('</think>', 1)[0] + '</think>' if '</think>' in text else ''
            call_id = f'call_{index}'
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"id": call_id, "type": "function",
                                             "function": {"name": name, "arguments": args}}]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps({"reward": event['reward'],
                                                    "observation": observation(event['after'])})})
        else:
            messages.append({"role": "assistant", "content": event['response']})
            messages.append({"role": "user", "content": "Invalid tool call; no requested action was executed. "
                             "Return one complete tool call.\n" + observation(event['after'])})
    return messages


def native_prompt_ids(tok, c, obs, history):
    budget = c.context_tokens - c.completion_tokens
    events = [h for h in history[-10:] if isinstance(h, dict)]
    current = dict(obs)
    for _ in range(40):
        messages = messages_for(c.system_prompt, current, events)
        ids = tok.apply_chat_template(messages, tools=TOOLS, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, preserve_thinking=True, return_dict=False)
        if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
            raise TypeError("Chat template must return a non-empty list of integer token IDs")
        if len(ids) <= budget:
            return ids
        if events:
            events = events[1:]  # discard whole call/result pairs, never orphan a result
        elif current['tree']:
            current['tree'] = current['tree'][:len(current['tree']) // 2]
        else:
            raise ValueError("System prompt, tools and goal exceed context_tokens")
    raise ValueError("Could not fit native tool prompt in context")
