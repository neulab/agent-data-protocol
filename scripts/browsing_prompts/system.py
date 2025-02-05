def get_system_message():
    with open('browsing_prompts/system_prompt.txt') as f: system_prompt = f.read()
    return system_prompt
