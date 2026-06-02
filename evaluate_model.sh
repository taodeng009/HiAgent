#vllm serve ./Qwen3.5-4B --host 0.0.0.0 --port 8000 --language-model-only --enforce-eager --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
export PROJECT_PATH='/home/dengtao/HiAgent'
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
MODEL="qwen3_5_4b_vllm_server"
export STEP=30
AGENT="ContextEfficientAgentV2"
AGENT="VanillaAgent"
export EVALTASK="alfworld"
python agentboard/eval_main.py \
    --cfg-path eval_configs/hiagent/blocksworld.yaml\
    --tasks alfworld \
    --model $MODEL  \
    --log_path ./logs/alfworld/hiagent/"smoke_test10_${AGENT}_${MODEL}_${STEP}"   \
    --project_name none \
    --baseline_dir ./data/baseline_results  \
    --max_num_steps $STEP     \
    --memory_size   100  \
    --agent $AGENT