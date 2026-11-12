import os

import pytest

from bulfm.config import Config
from bulfm.policy import prompt_ids
from bulfm.tools_protocol import TOOLS, messages_for


def event(valid=True):
    before = {'goal': 'Select date', 'tree': "[17] textbox readonly=True", 'error': ''}
    after = {**before, 'tree': "[111] link '28'"}
    return {'before': before, 'after': after, 'action': "click('17')", 'valid': valid,
            'response': "<think>Open picker</think><tool_call>ignored raw markup</tool_call>", 'reward': 0.0}


def test_native_schema_history_and_error_roles():
    e = event()
    messages = messages_for(Config().system_prompt, e['after'], [e])
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'tool']
    assert 'Python-style action' not in messages[0]['content']
    assert messages[2]['tool_calls'][0]['function'] == {'name': 'click', 'arguments': {'bid': '17'}}
    assert messages[3]['tool_call_id'] == messages[2]['tool_calls'][0]['id']
    assert "link '28'" in messages[3]['content']
    invalid = messages_for(Config().system_prompt, e['after'], [event(False)])
    assert invalid[-1]['role'] == 'user' and 'Invalid tool call' in invalid[-1]['content']
    assert not any(m.get('tool_calls') for m in invalid)


def test_native_prompt_prunes_whole_pairs_and_keeps_goal():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['tools'] == TOOLS and kwargs['enable_thinking'] is False
            self.messages = messages
            return [1] * (2000 if len(messages) > 2 else 200)
    tok = Tokenizer()
    e = event()
    prompt_ids(tok, Config(native_tools=True, context_tokens=512), e['after'], [e, e])
    assert [m['role'] for m in tok.messages] == ['system', 'user']
    assert 'Select date' in tok.messages[1]['content']
    assert "link '28'" in tok.messages[1]['content']


@pytest.mark.skipif(os.environ.get('BULFM_LFM3_TEST') != '1', reason='Real tokenizer opt-in')
def test_actual_lfm3_template_serializes_tools_and_history():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('LiquidAI/LFM3-1.7B', trust_remote_code=True)
    e = event()
    ids = prompt_ids(tok, Config(native_tools=True, context_tokens=8192), e['after'], [e])
    text = tok.decode(ids, skip_special_tokens=False)
    assert '<tools>' in text and '<function=click>' in text
    assert '<tool_response>' in text and "link '28'" in text
    assert text.endswith('<think></think>')
