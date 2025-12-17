"""
Math evaluation script with command line arguments and result saving.
Supports multiple datasets: AIME24, AIME25, MATH500, Olympiad, AMC23
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.utils import compute_pass_at_k


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate math agent on various datasets")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Model name")
    parser.add_argument("--base_url", type=str, default="http://localhost:30000/v1", help="vLLM server URL")
    parser.add_argument("--api_key", type=str, default="None", help="API key")
    parser.add_argument("--dataset", type=str, default="aime2024", 
                        choices=["aime2024", "aime2025", "math500", "olympiad", "amc23", "torl_math"],
                        help="Dataset to evaluate")
    parser.add_argument("--n_parallel", type=int, default=64, help="Number of parallel agents")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per problem for pass@k")
    parser.add_argument("--max_response_length", type=int, default=16384, help="Max response length")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    print(f"\n{'='*60}")
    print(f"Evaluation Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Base URL: {args.base_url}")
    print(f"  N Parallel: {args.n_parallel}")
    print(f"  N Samples: {args.n_samples}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Agent and environment configuration
    agent_args = {
        "tools": ["python"], 
        "parser_name": "qwen", 
        "system_prompt": "You are a math assistant that can write python to solve math problems."
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
    
    # Create execution engine
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
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    test_dataset = DatasetRegistry.load_dataset(args.dataset, "test")
    if test_dataset is None:
        print(f"Dataset {args.dataset} not found!")
        return
    
    print(f"Dataset loaded with {len(test_dataset)} problems")
    
    # Repeat for pass@k evaluation
    tasks = test_dataset.repeat(n=args.n_samples)
    print(f"Total trajectories to run: {len(tasks)}")
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    start_time = datetime.now()
    results = asyncio.run(engine.execute_tasks(tasks))
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Compute metrics
    metrics = compute_pass_at_k(results)
    
    # Prepare results data
    results_data = {
        "dataset": args.dataset,
        "model": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "config": {
            "n_parallel": args.n_parallel,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_response_length": args.max_response_length,
            "max_prompt_length": args.max_prompt_length,
        },
        "pass_at_1": metrics.get("pass_at_1", 0.0) if isinstance(metrics, dict) else 0.0,
        "pass_at_k": metrics.get("pass_at_k", 0.0) if isinstance(metrics, dict) else 0.0,
        "total_problems": len(test_dataset),
        "total_trajectories": len(tasks),
    }
    
    # Save scores
    scores_file = output_dir / f"{args.dataset}_scores.json"
    with open(scores_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nScores saved to: {scores_file}")
    
    # Save detailed inference results
    inference_file = output_dir / f"{args.dataset}_inference.jsonl"
    with open(inference_file, "w") as f:
        for i, result in enumerate(results):
            result_entry = {
                "trajectory_id": i,
                "problem_id": result.get("problem_id", i // args.n_samples),
                "reward": result.get("reward", 0.0),
                "done_reason": result.get("done_reason", "unknown"),
                "trajectory": result.get("trajectory", []),
            }
            f.write(json.dumps(result_entry) + "\n")
    print(f"Inference results saved to: {inference_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS for {args.dataset}:")
    print(f"  Pass@1: {results_data['pass_at_1']:.4f}")
    print(f"  Pass@k: {results_data['pass_at_k']:.4f}")
    print(f"  Total Problems: {results_data['total_problems']}")
    print(f"  Duration: {duration:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

