import os

_script_dir = os.path.dirname(os.path.realpath(__file__))

prompt_file = os.path.join(_script_dir, 'system_prompt.txt')
def get_system_message():
    with open(prompt_file, 'r') as f:
        system_prompt = f.read()
    return system_prompt
