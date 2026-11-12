"""Native LFM3 tool-call regressions from actual book-flight traces."""
import ast
import os

import pytest

from bulfm.environment import parse_action


def native(name, **parameters):
    arguments = ''.join(f'<parameter={key}>\n{value}\n</parameter>\n'
                        for key, value in parameters.items())
    return f'<tool_call>\n<function={name}>\n{arguments}</function>\n</tool_call>'


@pytest.mark.parametrize('city', [
    'OOK', 'MQT', 'Chadron, NE', 'Moses Lake, WA', 'Prescott, AZ', 'Peoria, IL',
    'Missoula, MT', 'Dillingham, AK', "O'Hare", 'Привет', '  keep spaces  ',
])
def test_observed_complete_reasoning_and_native_fill(city):
    source = "We need to fill the From field. Let's start.</think>" + native('fill', bid='19', text=city)
    action, valid = parse_action(source)
    assert valid
    parsed = ast.parse(action, mode='eval').body
    assert parsed.func.id == 'fill'
    assert [ast.literal_eval(arg) for arg in parsed.args] == ['19', city]


@pytest.mark.parametrize('source,expected', [
    (native('click', bid='27'), "click('27')"),
    (native('press', bid='19', key_comb='Enter'), "press('19', 'Enter')"),
    (native('press', bid='19', key='Enter'), "press('19', 'Enter')"),
    (native('select_option', bid='31', options='cheapest'), "select_option('31', 'cheapest')"),
    (native('scroll', delta_x=0, delta_y=500), 'scroll(0.0, 500.0)'),
    (native('noop'), 'noop()'),
])
def test_native_supported_actions(source, expected):
    assert parse_action(source) == (expected, True)


@pytest.mark.parametrize('source', [
    native('fill', bid='19'),
    native('fill', bid='19', text='OOK', unknown='bad'),
    native('click', bid='13') + native('click', bid='14'),
    native('fill', bid='19', text='OOK').replace('</tool_call>', ''),
    native('fill', bid='19', text='OOK').replace('<parameter=text>', '<parameter=bid>'),
    native('__import__', text='os'),
    native('scroll', delta_x=0, delta_y=10001),
    native('scroll', delta_x=0, delta_y='nan'),
    native('press', bid='19', key='Enter', keys='Escape'),
    '<think>' + native('click', bid='13') + '</think>',
    "We need to click. " + native('click', bid='13'),
    native('click', bid='13') + "\n__import__('os').system('id')",
])
def test_native_requires_one_complete_allowed_final_call(source):
    assert parse_action(source) == ('noop()', False)


@pytest.mark.skipif(os.environ.get('BULFM_REAL_BROWSER') != '1',
                    reason='Real browser opt-in')
def test_native_fill_changes_actual_book_flight_field():
    from bulfm.config import Config
    from bulfm.environment import Browser
    env = Browser(Config(task='miniwob.book-flight'))
    try:
        obs = env.reset(23)
        assert "[19] textbox 'From:'" in obs['tree']
        response = "We need to fill From.</think>" + native('fill', bid='19', text='OOK')
        action, valid = parse_action(response)
        assert valid
        observation, _, _ = env.step(action)
        assert not observation['error']
        assert env.env.unwrapped.page.get_by_role('textbox', name='From:', exact=True).input_value() == 'OOK'
    finally:
        env.close()
