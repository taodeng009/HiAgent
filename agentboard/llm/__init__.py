from .openai_gpt import OPENAI_GPT
from .azure_gpt import OPENAI_GPT_AZURE
from .claude import CLAUDE
from common.registry import registry
from .huggingface import HgModels
from .msal_gpt import MSAL_GPT

try:
    from .vllm import VLLM
except ModuleNotFoundError:
    VLLM = None

__all__ = [
    "OPENAI_GPT",
    "OPENAI_GPT_AZURE",
    "VLLM",
    "CLAUDE",
    "HgModels",
    "MSAL_GPT"
]





def load_llm(name, config):
    
    llm = registry.get_llm_class(name).from_config(config)
    
    return llm
    
