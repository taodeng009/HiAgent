from .vanilla_agent import VanillaAgent
from .react_agent import ReactAgent
from .custom_react import CustomReactAgent
from common.registry import registry
from .ours_agent import OurAgent
from .cme_final import ContextEfficientAgentV2
from .plugmem_agent import PlugMemContextEfficientAgent
from .gmemory_agent import GMemoryContextEfficientAgent
from .gmemory_vanilla_agent import GMemoryVanillaAgent
from .split_action_agent import SplitSubgoalActionContextEfficientAgent, GMemoryActionOnlyContextEfficientAgent

__all__ = ["VanillaAgent", "ReactAgent", "CustomReactAgent", "OurAgent", "ContextEfficientAgentV2", "PlugMemContextEfficientAgent", "GMemoryContextEfficientAgent", "GMemoryVanillaAgent", "SplitSubgoalActionContextEfficientAgent", "GMemoryActionOnlyContextEfficientAgent"]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    return agent
