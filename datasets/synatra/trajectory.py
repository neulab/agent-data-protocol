from enum import Enum
import json
import re

class ActionType(Enum):
    CLICK = 1
    TYPE = 2
    HOVER = 3
    PRESS = 4
    GOTO = 5
    GO_BACK = 6
    GO_FORWARD = 7
    NEW_TAB = 8
    CLOSE_TAB = 9
    SWITCH_TAB = 10
    STOP = 11
    SCROLL = 12

    def to_string(self):
        return self.name.lower()

class Action:
    def __init__(self, cot="", action_description="", action_type=None, axt_node_id=-1, subtask="", typed_string=""):
        self._cot = cot  
        self._action_description = action_description  
        self._action_type = action_type  
        self._axt_node_id = axt_node_id
        self._subtask = subtask
        self._typed_string = typed_string

    @property
    def cot(self):
        return self._cot

    @cot.setter
    def cot(self, value):
        self._cot = value

    @property
    def typed_string(self):
        return self._typed_string

    @typed_string.setter
    def typed_string(self, value):
        self._typed_string = value

    @property
    def subtask(self):
        return self._subtask

    @subtask.setter
    def subtask(self, value):
        self._subtask = value

    @property
    def action_description(self):
        return self._action_description

    @action_description.setter
    def action_description(self, value):
        self._action_description = value

    @property
    def action_type(self):
        return self._action_type

    @action_type.setter
    def action_type(self, value):
        if isinstance(value, ActionType):
            self._action_type = value
        else:
            raise ValueError("action_type must be an instance of ActionType")

    @property
    def axt_node_id(self):
        return self._axt_node_id

    @axt_node_id.setter
    def axt_node_id(self, value):
        if isinstance(value, int):
            self._axt_node_id = value
        else:
            raise ValueError("axt_node_id must be an integer")

    @classmethod
    def from_string(cls, action_str):
        action_str_lines = action_str.split('\n')
        action = cls()
        for line in action_str_lines:
            if line.strip().startswith('# sub-task '):
                action.subtask = line.strip()
            elif line.strip().startswith('# step') and not line.strip().startswith('# step summary:'):
                action.cot = line.strip()
            elif line.strip().startswith('# step summary:'):
                action.action_description = line.strip()
            elif '(' in line and ')' in line and '# step' not in line and '# step summary:' not in line and(
                ('click(' in line and '(element_id=' in line)
                or ('type(' in line and '(element_id=' in line)
                or ('hover(' in line and '(element_id=' in line)
                or 'key_press(' in line 
                or 'goto(' in line 
                or 'go_back(' in line 
                or 'go_forward(' in line 
                or 'new_tab(' in line 
                or 'close_tab(' in line 
                or 'switch_tab(' in line 
                or 'stop(' in line 
                or 'scroll(' in line
            ):

                response = line
                response = response.replace('element_id=','').replace('string=','')
                response = response.replace("'","")
                response = response.replace('"','')
                items = re.split('[(),]', response)
                items = [item.strip() for item in items]
                if 'click(' in line:
                    action.action_type = ActionType.CLICK
                    action.axt_node_id = int(items[1])
                elif 'type(' in line:
                    action.action_type = ActionType.TYPE
                    action.axt_node_id = int(items[1])
                    action.typed_string = str(items[2])
                elif 'hover(' in line:
                    action.action_type = ActionType.HOVER
                    action.axt_node_id = int(items[1])
                elif 'key_press(' in line:
                    action.action_type = ActionType.PRESS
                    action.typed_string = str(items[1])
                elif 'goto(' in line:
                    action.action_type = ActionType.GOTO
                    action.typed_string = str(items[1])
                elif 'go_back(' in line:
                    action.action_type = ActionType.GO_BACK
                elif 'go_forward(' in line:
                    action.action_type = ActionType.GO_FORWARD
                elif 'new_tab(' in line:
                    action.action_type = ActionType.NEW_TAB
                elif 'close_tab(' in line:
                    action.action_type = ActionType.CLOSE_TAB
                elif 'switch_tab(' in line:
                    action.action_type = ActionType.SWITCH_TAB
                elif 'stop(' in line:
                    action.action_type = ActionType.STOP
                    action.typed_string = str(items[1])
                elif 'scroll(' in line:
                    action.action_type = ActionType.SCROLL
                    action.typed_string = str(items[1])
                else:
                    raise(
                        f'Cannot parse action'
                    )
        return action

    def __str__(self):
        result_list = []
        if self._subtask:
            result_list.append(self._subtask)
        if self._cot:
            result_list.append(self._cot)
        action = self._action_type.to_string() + '('
        if self._axt_node_id != -1:
            action+=f'element_id="{self._axt_node_id}"'
        if self._axt_node_id != -1 and self._typed_string:
            action+=','
        if self._typed_string:
            if self._action_type == ActionType.TYPE:
                action+=f'string="{self._typed_string}"'
            else:
                action+=f"'{self._typed_string}'"
        action += ')'
        if action:
            result_list.append(action)
        if self._action_description:
            result_list.append(self._action_description)
        return '\n'.join(result_list)

class Trajectory:
    def __init__(self, entry=None, next_action=Action(), history="", obs="", objective="", website=""):
        if entry:
            prompt_string = entry.prompt
            lines = prompt_string.split('\n')
            extracted_parts = {
                'objective': [],
                'past_actions': [],
                'website': [],
                'observation': []
            }
            
            current_category = None
            for line in lines:
                if line.startswith('# website'):
                    current_category = 'website'
                elif line.startswith('# objective'):
                    current_category = 'objective'
                elif line.startswith('# observation of the current web page'):
                    current_category = 'observation'
                elif line.startswith('# past actions'):
                    current_category = 'past_actions'
                elif current_category:
                    extracted_parts[current_category].append(line)
            for category in extracted_parts:
                extracted_parts[category] = '\n'.join(extracted_parts[category])
            extracted_parts['objective'] = re.search(r'objective = "(.*?)"\n', extracted_parts['objective'], re.DOTALL).group(1)
            extracted_parts['website'] = re.search(r'website = "(.*?)"', extracted_parts['website']).group(1)
            extracted_parts['observation'] = re.search(r'observation = """([\s\S]*?)"""\n', extracted_parts['observation']).group(1)
            self._next_action = Action.from_string(entry.response)

            # history_lines  = extracted_parts['past_actions'].split('\n')[1:]
            # st = ed = 0
            # history_list = []
            # while ed < len(history_lines):
            #     if '(' in history_lines[ed] and ')' in history_lines[ed] and '# step' not in history_lines[ed] and '# step summary:' not in history_lines[ed] and(
            #         ('click(' in history_lines[ed] and '(element_id=' in history_lines[ed])
            #         or ('type(' in history_lines[ed] and '(element_id=' in line)
            #         or ('hover(' in history_lines[ed] and '(element_id=' in history_lines[ed])
            #         or 'key_press(' in history_lines[ed] 
            #         or 'goto(' in history_lines[ed] 
            #         or 'go_back(' in history_lines[ed] 
            #         or 'go_forward(' in history_lines[ed] 
            #         or 'new_tab(' in history_lines[ed] 
            #         or 'close_tab(' in history_lines[ed] 
            #         or 'switch_tab(' in history_lines[ed] 
            #         or 'stop(' in history_lines[ed] 
            #         or 'scroll(' in history_lines[ed]
            #     ):
            #         try:
            #             action = Action.from_string('\n'.join(history_lines[st:ed+1]))
            #             action.action_description = action.cot
            #             action.cot = ''
            #             history_list.append(action)
            #         except:
            #             pass
            #         st = ed+1
            #     ed+=1
            # extracted_parts['past_actions'] = history_list

            self._history = extracted_parts['past_actions']
            self._obs = extracted_parts['observation']
            self._objective = extracted_parts['objective']
            self._website = extracted_parts['website']
        else:
            self._next_action = next_action
            # self._history = history if history is not None else []
            self._history = history
            self._objective = objective
            self._obs = obs
            self._website = website

    @property
    def next_action(self):
        return self._next_action

    @next_action.setter
    def next_action(self, value):
        if isinstance(value, Action):
            self._next_action = value
        else:
            raise ValueError("next_action must be an instance of Action")

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        # if isinstance(value, list) and all(isinstance(item, Action) for item in value):
        #     self._history = value
        # else:
        #     raise ValueError("history must be a list of Action instances")
        if isinstance(value, str):
            self._history = value
        else:
            raise ValueError("history must be a string")

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, value):
        if isinstance(value, str):
            self._obs = value
        else:
            raise ValueError("obs must be a string")

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if isinstance(value, str):
            self._objective = value
        else:
            raise ValueError("objective must be a string")

    @property
    def website(self):
        return self._website

    @website.setter
    def website(self, value):
        if isinstance(value, str):
            self._website = value
        else:
            raise ValueError("website must be a string")

        
