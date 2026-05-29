import pdb

from agents.base_agent import BaseAgent
from common.registry import registry
# from rouge import Rouge
import json
import random
import re
import os

from .summarize import TrajectorySummarizer


def extract_numbers(action_string):
    matches = re.findall(r'retrieve\((\d+(?:, \d+)*)\)', action_string)
    if matches:
        return [int(id) for id in matches[0].split(', ')]
    else:
        return []

@registry.register_agent("ContextEfficientAgentV2")
class ContextEfficientAgentV2(
    BaseAgent):  # the agent should receive goal, state and action, then return the next state
    def __init__(self,
                 llm_model,
                 memory_size=100,
                 # set this to a very large number if you want to keep all history till context length limit
                 examples=[],
                 instruction="",
                 init_prompt_path=None,
                 system_message="You are a helpful assistant.",
                 need_goal=False,
                 check_actions=None,
                 check_inventory=None,
                 use_parser=True,
                 enable_retrieve_instruction=True,
                 check_actions_prompt_mode="strict",
                 ):
        super().__init__()
        self.use_parser = use_parser
        self.llm_model = llm_model
        self.memory_size = memory_size
        self.goal = None
        self.init_obs = None
        if init_prompt_path is not None:  # load from file
            self.init_prompt_dict = json.load(open(init_prompt_path, 'r'))
            self.instruction = self.init_prompt_dict["instruction"]
            self.examples = self.init_prompt_dict["examples"]
        else:

            self.instruction = instruction
            self.examples = examples

            # self.reset(goal, init_obs)
            self.init_prompt_dict = {
                "examples": examples,
                "instruction": instruction,
                "system_msg": system_message
            }

        self.max_context_length = self.llm_model.context_length
        self.need_goal = need_goal
        self.check_actions = check_actions
        self.check_inventory = check_inventory
        self.enable_retrieve_instruction = enable_retrieve_instruction
        self.check_actions_prompt_mode = check_actions_prompt_mode

        self.example_prompt = None

        if "claude" in self.llm_model.engine:
            self.split = self.llm_model.xml_split
        else:
            self.split = {"example": [""],
                          "text": [""],
                          "rule": [""],
                          "system_msg": [""],
                          "instruction": [""],
                          "goal": [""]}
        
        self.subgoal_idx = []

    def get_example_prompt(self): #return the prompt for an interaction turn
        return self.example_prompt
    
    def log_example_prompt(self, prompt):
        self.example_prompt = prompt

    def log_example_prompt_subgoal(self, prompt):
        self.example_prompt = prompt
    
    def log_example_prompt_action(self, prompt):
        self.example_prompt = f'```subgoal\n{self.example_prompt}\n```\n```action\n{prompt}\n'

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        self.memory = [[("Action", init_act), ('Observation', self.init_obs)]] if init_act \
            else [
            [('Observation', self.init_obs)]]  # list of [('State', "xxx"), ('Action', "xxx"), ...]
        self.steps = 0
        self.done = False

    def update(self, action, state):
        self.steps += 1

        # self.memory.append(("Action", action))
        # self.memory.append(("Observation", state))
        self.memory.append([("Action", action), ('Observation', state)])

    def make_prompt(self, need_goal=False, check_actions="check valid actions", check_inventory="inventory", system_message=''):
        def vanilla_serialize_history(history):
            res = []
            for item in history:
                for _ in item:
                    res.append( _[0] + ": " + _[1])
            return '\n'.join(res)

        def serialize_history(history):
            self.task = os.environ.get('EVALTASK')
            # if self.task in ['gripper', 'blocksworld']:
            if any([_ in self.task for _ in ['gripper', 'blocksworld']]):
                summarization = False    # For gripper and blocksworld, set to False.
            else:
                summarization = True
            # ommit_prefix = 'Subgoal is satisfied, and the process is ommited. '
            ommit_prefix = 'Subgoal is satisfied. '
            # locate last subgoal
            subgoal_index_list = []
            keep_subgoal_index_list = [_-1 for _ in self.subgoal_idx]
            for i in range(0, len(history)):
                item = history[i]
                if item[0][0] == 'Subgoal': # last subgoal
                    subgoal_index_list.append(i)
            if len(subgoal_index_list) <= 1:
                return vanilla_serialize_history(history)
            final_subgoal = subgoal_index_list[-1]
            new_history = history[:subgoal_index_list[0]]
            for i in range(0, len(subgoal_index_list)-1):
                if i in keep_subgoal_index_list:
                    new_history += history[subgoal_index_list[i]:subgoal_index_list[i+1]]
                    continue
                index = subgoal_index_list[i]
                obs_index = subgoal_index_list[i+1] - 1
                if not summarization:   # No summarization. Under full observable environment
                    subgoal = history[index][0]
                    _ = subgoal[0]
                    subgoal = (f'{i+1} {_}', subgoal[1])
                    # new_history.append([subgoal, ("Observation", ommit_prefix + history[obs_index][1][1])])
                    new_history.append([subgoal, ("Observation", history[obs_index][1][1])])
                else:   # Using the summarizer
                    summarizer = TrajectorySummarizer(self.llm_model)
                    subgoal = history[index][0]
                    trajectory = history[index+1:obs_index+1]
                    # remove check valid actions
                    trajectory = [pair for pair in trajectory if pair[0][0] != 'Action' or 'check valid' not in pair[0][1]]
                    # if len(trajectory) == 1:
                    #     pair = trajectory[0]
                    #     if pair[0][0] == 'Observation':
                    #         summary = pair[0][1]
                    #     else:
                    #         summary = pair[1][1]
                    # else:
                    summary = summarizer.generate_summary([trajectory], [subgoal])[0]
                    # reformat subgoal
                    subgoal = history[index][0]
                    _ = subgoal[0]
                    subgoal = (f'{i+1} {_}', subgoal[1])
                    new_history.append([subgoal, ("Observation", summary)])

            # new_history += history[final_subgoal:]
            # add number of last subgoal
            # - = []
            subgoal = history[final_subgoal][0]
            _ = subgoal[0]
            subgoal = (f'{len(subgoal_index_list)} {_}', subgoal[1])
            _ = [[subgoal]] + history[final_subgoal+1:]
            new_history += _
            return vanilla_serialize_history(new_history)

        query = ""
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal, while an action is a specific step executed in the environment. When there is an unfinished subgoal, you need to output an action to continue completing this subgoal in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed in the following format: \"Subgoal: {subgoal}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed in the following format: \"Subgoal: {subgoal}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". You cannot output two subgoals consecutively."
        # _ = "\nNote: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". You cannot output two subgoals consecutively. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id)\" to obtain the detailed trajectory information. You should use this method judiciously for token efficiency."
        # remove the constrains of retrieve
        #         _ = """
        # Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. 
        # When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {action}\". 
        # When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {subgoal}\\nAction: {action}\". 
        # Hints:
        # 1. You cannot output two subgoals consecutively. 
        # 2. Subgoal must be one line of text and does not print any newline characters. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, you can use Action: \"retrieve(subgoal_id_1, subgoal_id_2, ...)\" to obtain the detailed trajectory information.
        # """
        retrieve_instruction = (
            '4. Detailed trajectory information (action-observation pair) of previously satisfied subgoals will be hidden for context efficiency. '
            'If you believe that the detailed trajectory information of a particular subgoal is crucial for the current subgoal, '
            'you can use Action: "retrieve(subgoal_id_1, subgoal_id_2, ...)" to obtain the detailed trajectory information.\n'
            if self.enable_retrieve_instruction
            else ""
        )
        if self.check_actions_prompt_mode == "strict":
            check_actions_rule = (
                '3. Each subgoal must be followed by at least one valid action. '
                'If the current action fails, you need to execute "check valid actions" to get a list of valid actions and select one from the list.\n'
            )
        elif self.check_actions_prompt_mode == "soft":
            check_actions_rule = (
                '3. Each subgoal must be followed by at least one valid action. '
                'After two consecutive "Nothing happens." observations, use "check valid actions" and select the next action from the listed valid actions.\n'
            )
        elif self.check_actions_prompt_mode == "soft2":
            check_actions_rule = (
                '3. Each subgoal must be followed by at least one valid action. '
                'If two consecutive actions fail, you need to execute "check valid actions" to get a list of valid actions and select one from the list.\n'
            )
        elif self.check_actions_prompt_mode == "none":
            check_actions_rule = '3. Each subgoal must be followed by at least one valid action.\n'
        else:
            raise ValueError(f"Unsupported check_actions_prompt_mode: {self.check_actions_prompt_mode}")
        _ = f"""
Note: A subgoal is a milestone goal that you need to complete in order to achieve the final goal. 
When there is an unfinished subgoal, you need to ground the given subgoal to corresponding executable actions for solving the given task in the following format: \"Action: {{action}}\". 
When there is no current subgoal or you believe the previous subgoal has been completed (based on past actions and observations), you need to output the next subgoal to be completed and its first action in the following format: \"Subgoal: {{subgoal}}\\nAction: {{action}}\". 
Instructions:
1. You cannot output two subgoals consecutively. 
2. Subgoal must be one line of text and does not print any newline characters. 
{check_actions_rule.rstrip()}
{retrieve_instruction.rstrip()}
        """
       

        if _ not in self.instruction:
            self.instruction += _
        query += self.split["instruction"][0] + self.instruction + self.split["instruction"][-1]

        if isinstance(self.examples, str):
            self.examples = [self.examples]

        if len(self.examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in self.examples:
                query += example + "\n"
            query += self.split["example"][-1]
        if need_goal:
            query += self.split["goal"][0] + "You should perform actions to accomplish the goal: " + self.goal + "\n" + \
                     self.split["goal"][-1]
        if check_actions is not None:
            query += "You should use the following commands for help when your action cannot be understood: " + check_actions + "\n"
        if check_inventory is not None:
            query += "You should use the following commands for help when your action cannot be understood: inventory\n"

        history = self.memory[-self.memory_size:]
        input_prompt = query + serialize_history(history)

        input_prompt += "\nAction: " if self.memory[-1][0][0] == 'Subgoal' else ""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt}
        ]
        num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        while num_of_tokens > self.max_context_length - self.llm_model.max_tokens:
            history = history[1:]
            input_prompt = query + serialize_history(history)
            # input_prompt += "\nAction: "
            input_prompt += "\nAction: " if self.memory[-1][0][0] == 'Subgoal' else ""
            # input_prompt += "\nPlease enter your action:"
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt}
            ]
            num_of_tokens = self.llm_model.num_tokens_from_messages(messages)
        print(f'------------[Prompt Start]-----------\n{input_prompt}\n----------[Prompt END]------------')
        return input_prompt

    def action_parser_for_special_llms(self, action):
        
        '''
        This function is used to parse the action for special llms, e.g. codellama-13b, codellama-34b, llama, lemur, vicuna, etc.
        These llms often struggle to generate the format of the action correctly, so we need to parse the action to make it executable.
        '''
        
        origin_action = action
        if 'action' in action.lower():
            action_temp = action.split('\n')
            for act in action_temp:
                if "next action" in act and ':' in act: # zzh: in Claude will return "Here is the next action to take:"
                    idx = action_temp.index(act)
                    while idx + 1 < len(action_temp):
                        if action_temp[idx + 1]:
                            action = action_temp[idx + 1]
                            break
                        idx += 1
                if act.split(':')[0].lower().endswith('with action input'): # chang: in case parse tool output
                    action = act
                    break
                if 'action' in act.lower() and ':' in act:
                    action_temp = ':'.join(act.split(':')[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if 'action' in act.lower() and 'is to' in act:
                    action_temp = act.split('is to')[1]
                    if action_temp != "":
                        action = action_temp
                        break
                        
        # if action.strip() == "":
        #     action = origin_action.split('\n')[0]   # temperary comment this line for codellama
        action = action.strip()
        action = action.strip("'/")
        action = action.split('\n')[0]
        return action

    def run(self, init_prompt_dict=None):
        # note that these configs are originally provided when initialized, but you can choose to override them here with parameters
        if init_prompt_dict is not None:
            self.init_prompt_dict = init_prompt_dict
            self.instruction = init_prompt_dict['instruction']
            self.examples = init_prompt_dict['examples']
        system_message = self.init_prompt_dict['system_msg']
        input_prompt = self.make_prompt(need_goal=self.need_goal,
                                        check_actions=self.check_actions,
                                        check_inventory=self.check_inventory,
                                        system_message=system_message)
        
        self.log_example_prompt(input_prompt)
        
        success, action = self.llm_model.generate(system_message, input_prompt)
        print(f'-------------GPT Response---------\n{action}\n---------------[END]------------')
        if success:
            # action = action.split('\n')[0]
            is_action = 'Subgoal' not in action
            # is_action = action.startswith('Action') or 'Subgoal' not in action
            if not is_action:
                # match = re.search(r"Subgoal:(.*?)(?=\n)", action)
                # if match:
                #     subgoal = match.group(1).strip()
                # action = action.replace(f'Subgoal: {subgoal}', '')
                subgoal = action.split('\n')[0]
                subgoal = subgoal.replace('Subgoal:', '')
                self.subgoal_idx = []
                self.memory.append([("Subgoal", subgoal)])
                action = '\n'.join(action.split('\n')[1:])
            # print('original output', action)
            # print(self.use_parser)
            # if is_action:
            if self.use_parser:
                action = self.action_parser_for_special_llms(action)
                # print('after parse', action)    
            if 'retrieve(' in action.lower():   # retrieve function is called
                action = action.lower()
                numbers = extract_numbers(action)
                # self.subgoal_idx.append(number)
                self.subgoal_idx += numbers
                self.run(init_prompt_dict)           
            # else:   # subgoal
            #     subgoal = action.replace('Subgoal:', '')
            #     self.memory.append([("Subgoal", subgoal)])
            #     return self.run(init_prompt_dict)
        return success, action

    @classmethod
    def from_config(cls, llm_model, config):
        memory_size = config.get("memory_size", 100)
        instruction = config.get("instruction", "")
        examples = config.get("examples", [])
        init_prompt_path = config.get("init_prompt_path", None)
        system_message = config.get("system_message", "You are a helpful assistant.")
        check_actions = config.get("check_actions", None)
        check_inventory = config.get("check_inventory", None)
        use_parser = config.get("use_parser", True)
        need_goal = config.get("need_goal", False)
        enable_retrieve_instruction = config.get("enable_retrieve_instruction", True)
        check_actions_prompt_mode = config.get("check_actions_prompt_mode", "strict")
        return cls(llm_model, memory_size, examples, instruction, init_prompt_path, system_message, 
                   need_goal, check_actions, check_inventory, use_parser, enable_retrieve_instruction,
                   check_actions_prompt_mode)
