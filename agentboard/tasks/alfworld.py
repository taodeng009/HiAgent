import json
from agents import load_agent
from environment import load_environment
from llm import load_llm
from common.registry import registry
import copy

from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
logger = AgentLogger(__name__)


from .base_task import BaseTask


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}



@registry.register_task("alfworld")
class Evalalfworld(BaseTask):
    def __init__(self,
                 llm_config=None,
                 agent_name='agent_name',
                 max_num_steps=30,
                 num_exams=134,
                 init_prompt_path='prompts/alfworld_base.json',
                 agent_config=None,
                 env_config=None,
                 llm = None,
                 baseline_dir = None,
                 log_path = None
                 ):
        
        super().__init__()
        
        ####################  initialize llm and agent ##################
        if llm is None:
            llm = load_llm(llm_config.get("name", "gpt"), llm_config)
        self.agent = load_agent(agent_name, agent_config, llm)
        #################################################################
        
        with open(init_prompt_path, 'r') as f:
            self.prompts = json.load(f)
        self.env_cfg = env_config
        self.max_num_steps = max_num_steps
        self.num_exams = num_exams
        
        self.baseline_dir = baseline_dir
        
        
        self.agentboard = TaskLogger(task_name="alfworld", log_path=log_path, max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir)

    def parseAction(self, action):
        action = action.strip()
        if "put" in action:
            if " in " in action:
                action = action.replace(" in ", ' in/on ')
            elif " on " in action:
                action = action.replace(" on ", ' in/on ')
        if action.endswith('.'):
            action = action[:-1].strip()
        return action

    def _remember_current_task(self, index, task_type, success, progress_rate=None, score_change_record=None):
        if not hasattr(self.agent, "remember_current_task"):
            return
        try:
            self.agent.remember_current_task(
                task_type=task_type,
                success=success,
                progress_rate=progress_rate,
                score_change_record=score_change_record,
                metadata={
                    "env": "alfworld",
                    "difficulty": self.env.difficulty,
                    "task_name": task_type,
                    "index": index,
                },
            )
        except Exception as exc:
            logger.warning("remember_current_task failed: {}".format(exc))

    def _get_example_prompt(self):
        try:
            return self.agent.get_example_prompt()
        except Exception:
            return None

    def _get_agent_diagnostics(self, action_stats=None):
        get_diagnostics = getattr(self.agent, "get_diagnostics", None)
        if not callable(get_diagnostics):
            return None
        try:
            diagnostics = get_diagnostics()
        except Exception as exc:
            logger.warning("get_diagnostics failed: {}".format(exc))
            return None
        if diagnostics is None:
            return None
        if action_stats is None or not isinstance(diagnostics, dict):
            return diagnostics

        diagnostics = copy.deepcopy(diagnostics)
        diagnostics.setdefault("invalid_action_count", action_stats.get("invalid_action_count", 0))
        diagnostics.setdefault("nothing_happens_count", action_stats.get("nothing_happens_count", 0))
        diagnostics.setdefault("check_valid_actions_count", action_stats.get("check_valid_actions_count", 0))

        step_updates = action_stats.get("steps", [])
        steps = diagnostics.get("steps")
        if isinstance(steps, list):
            for idx, step_update in enumerate(step_updates):
                step_id = step_update.get("step", idx)
                matched_step = None
                for step in steps:
                    if isinstance(step, dict) and step.get("step") == step_id:
                        matched_step = step
                        break
                if matched_step is None and idx < len(steps) and isinstance(steps[idx], dict):
                    matched_step = steps[idx]
                if matched_step is not None:
                    matched_step.update(step_update)
            diagnostics["steps"] = steps
        diagnostics["alfworld_action_stats"] = copy.deepcopy(action_stats)
        return diagnostics

    def evaluate_env(self,  index, ob='', examples=None, task_type=None):

        init_ob = ob.split('\n')[0]
        goal = ob.split('\n')[1].split("Your task is to:")[1].strip()

        if task_type is not None:
            if hasattr(self.agent, "set_current_task_type"):
                self.agent.set_current_task_type(task_type)
            else:
                self.agent.current_task_type = task_type
        self.agent.reset(goal=goal, init_obs=init_ob)
        logger.goal("Example {} | Goal: {}".format(index, self.agent.goal))
        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict['examples'] = examples
        reward = 0.
        last_reward = 0.
        done = False
        grounding_acc_count = 0
        score_change_record = []
        logger.info("Step {:02} - Message: {}".format(0, init_ob))
        
        trajectory = []
        trajectory.append({"Goal":goal, "id":0})
        trajectory.append({"Observation":init_ob, "id":0})   
        action_stats = {
            "invalid_action_count": 0,
            "nothing_happens_count": 0,
            "check_valid_actions_count": 0,
            "steps": [],
        }
        
        for i in range(0, self.max_num_steps):
            success, agent_returned_action = self.agent.run(init_prompt_dict=init_prompt_dict)
            
            if not success:
                break
            
            action = self.parseAction(agent_returned_action)
            is_valid_action = action in self.env.get_action_space()
            if not is_valid_action:
                action_stats["invalid_action_count"] += 1
            if action == "check valid actions":
                action_stats["check_valid_actions_count"] += 1
            if action in self.env.get_action_space():
                grounding_acc_count += 1.0
            
            logger.info("Step {:02} - Action: {}".format(i, action))
            trajectory.append({"Action":action, "id":i})
            
            observation, reward, done, info = self.env.step(action)
            nothing_happens = observation.strip() == "Nothing happens."
            if nothing_happens:
                action_stats["nothing_happens_count"] += 1
            action_stats["steps"].append({
                "step": i,
                "agent_returned_action": agent_returned_action,
                "executed_action": action,
                "is_valid_action": is_valid_action,
                "nothing_happens": nothing_happens,
            })
            logger.info("Step {:02} - Observation: {}".format(i, observation))

            if "Task accomplished!" in observation and reward < 1.0:
                raise Exception("Task accomplished error")
            
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            
            trajectory.append({"Observation":observation, "id":i})
            trajectory.append({"Progress Rate":reward, "id":i})
            
            #print(f'Step: {str(i)} Action: {action}\nObservation: {observation}')
            #print(f"reward: {reward}, isdone: {done}")
            
            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            self.agent.update(action=action, state=observation)
            if done:
                game_name = self.env.cur_task_name.split('/')[0]
                env_details = {"task_name": game_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
                example_prompt = self._get_example_prompt()
                agent_diagnostics = self._get_agent_diagnostics(action_stats)
                self.agentboard.log_example(index, True, reward, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory, example_prompt, agent_diagnostics)
                self._remember_current_task(index, game_name, done, reward, score_change_record)

                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        
        game_name = self.env.cur_task_name.split('/')[0]
        env_details = {"task_name": game_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
        

        progress_rate = reward

        example_prompt = self._get_example_prompt()
        agent_diagnostics = self._get_agent_diagnostics(action_stats)
        self.agentboard.log_example(index, done, progress_rate, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory, example_prompt, agent_diagnostics)
        self._remember_current_task(index, game_name, done, progress_rate, score_change_record)

        return progress_rate, done, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        self.env = load_environment('alfworld', self.env_cfg)
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        for id in range(self.num_exams):

            ob, info = self.env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            #sub_goal = selected_obs[name]
            difficulties.append(self.env.difficulty)

            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    examples = "".join(self.prompts['examples'][v])
                    task_type = {
                        "put": "place",
                        "examine": "look",
                    }.get(v, v)
                    score, is_done, grounding_acc, score_change_record, steps = self.evaluate_env(
                        ob=ob,
                        examples=examples,
                        index=id,
                        task_type=task_type,
                    )
                    if is_done:
                        srs.append(1.0)
                    else:
                        srs.append(0.0)
                    scores.append(score)
                    grounding_accs.append(grounding_acc)
                    score_state_records.append(score_change_record)
                    #print(f"the {i}th task: reward: {score}")
                    logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(id, is_done, score, steps))

        sr = sum(srs) * 1.0 / len(srs)
        pr = sum(scores) * 1.0 / len(scores)
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        hard_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if len(hard_sr) > 0 else 0

        hard_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if len(hard_pr) > 0 else 0

        easy_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if len(easy_sr) > 0 else 0

        easy_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if len(easy_pr) > 0 else 0
                    
        
        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)

        return  srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    def _grounding_fn(self, action):

        if action not in self.env.GetValidActions():
            print(f"The wrong action is: {action}")
            return "check valid actions"
        else:
            return action

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm = None  
                    ):

        agent_name = agent_config.get("name", "GPTAgent")
        init_prompt_path = agent_config.get("init_prompt_path", 'prompts/alfworld_in_context_learning.json') 
        max_num_steps = run_config.get("max_num_steps", 30)
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        # wandb = run_config.get("wandb", False)
        num_exams = run_config.get("num_exam", 134)
        log_path = run_config.get("log_path", None)
        return cls(
                   llm_config=llm_config,
                   agent_name=agent_name,
                   max_num_steps=max_num_steps,
                   num_exams=num_exams,
                   init_prompt_path=init_prompt_path,
                   agent_config=agent_config,
                   env_config=env_config,
                   llm = llm,
                   baseline_dir = baseline_dir,
                   log_path = log_path
                   )
