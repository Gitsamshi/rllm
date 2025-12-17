"""
Math evaluation script with command line arguments.
Simple version for quick single-dataset evaluation.
"""

import argparse
import asyncio
import os

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate math agent on a dataset")
    parser.add_argument("--model_name", type=str, default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B"), 
                        help="Model name (or set MODEL_NAME env var)")
    parser.add_argument("--base_url", type=str, default=os.environ.get("BASE_URL", "http://localhost:30000/v1"), 
                        help="vLLM server URL (or set BASE_URL env var)")
    parser.add_argument("--api_key", type=str, default="None", help="API key")
    parser.add_argument("--dataset", type=str, default="aime2024", help="Dataset to evaluate")
    parser.add_argument("--n_parallel", type=int, default=int(os.environ.get("N_PARALLEL", "64")), 
                        help="Number of parallel agents")
    parser.add_argument("--n_samples", type=int, default=int(os.environ.get("N_SAMPLES", "8")), 
                        help="Number of samples per problem for pass@k")
    parser.add_argument("--max_response_length", type=int, default=16384, help="Max response length")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--parser_name", type=str, default="qwen", 
                        choices=["qwen", "llama", "default"], help="Parser name for agent")
    parser.add_argument("--system_prompt", type=str, 
                        default="You are a math assistant that can write python to solve math problems.",
                        help="System prompt for the agent")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    print(f"\n{'='*60}")
    print(f"Math Evaluation Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Base URL: {args.base_url}")
    print(f"  Dataset: {args.dataset}")
    print(f"  N Parallel: {args.n_parallel}")
    print(f"  N Samples: {args.n_samples}")
    print(f"  Parser: {args.parser_name}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    agent_args = {
        "tools": ["python"], 
        "parser_name": args.parser_name, 
        "system_prompt": args.system_prompt
    }
    env_args = {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    }

    sampling_params = {
        "temperature": args.temperature, 
        "top_p": args.top_p, 
        "model": args.model_name
    }

    engine = AgentExecutionEngine(
        agent_class=ToolAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args={"base_url": args.base_url, "api_key": args.api_key},
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        n_parallel_agents=args.n_parallel,
    )

    print(f"Loading dataset: {args.dataset}")
    test_dataset = DatasetRegistry.load_dataset(args.dataset, "test")
    if test_dataset is None:
        print("Dataset not found, preparing dataset...")
        from prepare_math_data import prepare_math_data
        _, test_dataset = prepare_math_data()

    print(f"Dataset loaded with {len(test_dataset)} problems")
    
    tasks = test_dataset.repeat(n=args.n_samples)  # repeat to evaluate pass@k
    print(f"Total trajectories to run: {len(tasks)}")

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)
