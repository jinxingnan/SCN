"""
Run model inference once and save complete beam results.

This script performs a single model inference pass and saves all beam paths
with their scores, answers, and states. The saved results can be used by
compute_metrics.py to calculate metrics with different thresholds without
re-running inference.

Usage:
    python run_inference.py \
        --checkpoint_dir output/scn_ckp_xxx \
        --eval_file datasets/RoG-webqsp-test.jsonl \
        --output_file results/inference_raw.json
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from scn.config import SCNConfig
from scn.model import SCNReasoner
from scn.inference import BeamSearchInference
from embeddings.kge_loader import KGELoader
from data.dataset import SCNDataset


def load_model(
    checkpoint_dir: str,
    config: SCNConfig,
    kge_loader: KGELoader,
    device: torch.device,
) -> SCNReasoner:
    """Load trained SCN model from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create model
    model = SCNReasoner(
        config=config,
        kge_loader=kge_loader,
        device=device,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model_state_dict"]
    
    model.input_projector.load_state_dict(model_state["input_projector"])
    model.output_projector.load_state_dict(model_state["output_projector"])
    model.candidate_builder.load_state_dict(model_state["candidate_builder"])
    
    if config.use_lora:
        lora_path = checkpoint_dir / "lora_best"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        model.llm.load_lora_weights(str(lora_path))
    
    print(f"Model loaded from {checkpoint_dir}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    return model


@torch.no_grad()
def run_inference(
    model: SCNReasoner,
    dataset: SCNDataset,
    beam_size: int = 5,
    max_hops: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run inference on dataset and collect all beam paths.
    
    Returns:
        Dict with inference results (no threshold-dependent metrics)
    """
    model.eval()
    
    inference_engine = BeamSearchInference(
        model=model,
        beam_size=beam_size,
        max_hops=max_hops,
    )
    
    samples_data = []
    
    iterator = tqdm(dataset.samples, desc="Running inference") if verbose else dataset.samples
    
    for sample in iterator:
        # Run beam search
        result = inference_engine.search(
            question=sample.question,
            topic_entities=sample.topic_entities,
            subgraph=sample.subgraph_triples,
        )
        
        # Collect beam paths info
        beam_paths = []
        for i, beam_path in enumerate(result.all_paths[:beam_size]):
            path_info = {
                "rank": i + 1,
                "answer": beam_path.get_answer(),
                "score": float(beam_path.score),  # Convert to float for JSON
                "is_stopped": bool(beam_path.is_stopped),
                "path_length": len(beam_path.path),
                "path": [[h, r, t] for h, r, t in beam_path.path] if beam_path.path else [],
            }
            beam_paths.append(path_info)
        
        # Collect all valid gold paths
        all_gold_paths = [sample.gold_path] if sample.gold_path else []
        if sample.alternative_paths:
            all_gold_paths.extend(sample.alternative_paths)

        # Save sample data
        sample_data = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "topic_entities": sample.topic_entities,
            "gold_answers": sample.all_answers,
            "gold_path": [[h, r, t] for h, r, t in sample.gold_path] if sample.gold_path else None,
            "all_gold_paths": [[[h, r, t] for h, r, t in path] for path in all_gold_paths],
            "num_gold_paths": len(all_gold_paths),
            "beam_paths": beam_paths,
            "num_steps": result.num_steps,
        }
        
        samples_data.append(sample_data)
    
    return samples_data


def main():
    parser = argparse.ArgumentParser(description="Run SCN inference and save beam results")
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--kge_embeddings_dir",
        type=str,
        default=None,
        help="Path to KGE embeddings directory",
    )
    parser.add_argument(
        "--kge_mapping_dir",
        type=str,
        default=None,
        help="Path to KGE mapping directory",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for inference",
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=5,
        help="Maximum reasoning hops",
    )
    parser.add_argument(
        "--inference_temperature",
        type=float,
        default=None,
        help="Override model temperature during inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for inference results (JSON)",
    )
    
    args = parser.parse_args()
    
    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = SCNConfig.from_yaml(str(config_path))
    print(f"Config loaded from {config_path}")
    
    # Override with command line args
    kge_embeddings_dir = args.kge_embeddings_dir or config.kge_embeddings_dir
    kge_mapping_dir = args.kge_mapping_dir or config.kge_mapping_dir
    eval_file = args.eval_file or config.val_file
    
    if not kge_embeddings_dir or not kge_mapping_dir or not eval_file:
        raise ValueError("Missing required paths. Specify via command line or config.")
    
    # Load KGE embeddings
    print(f"Loading KGE embeddings from {kge_embeddings_dir}...")
    kge_loader = KGELoader(
        embeddings_dir=kge_embeddings_dir,
        mapping_dir=kge_mapping_dir,
        device=device,
    )
    
    # Load model
    print("Loading model...")
    model = load_model(
        checkpoint_dir=str(checkpoint_dir),
        config=config,
        kge_loader=kge_loader,
        device=device,
    )

    if args.inference_temperature is not None:
        model.temperature = args.inference_temperature
        print(f"Override inference temperature: {model.temperature}")
    
    # Load dataset
    print(f"Loading dataset from {eval_file}...")
    dataset = SCNDataset(data_file=eval_file, skip_no_paths=False)
    
    # Run inference
    print(f"\n{'='*60}")
    print(f"Running inference on {len(dataset)} samples...")
    print(f"Beam size: {args.beam_size}")
    print(f"Max hops: {args.max_hops}")
    print(f"{'='*60}\n")
    
    samples_data = run_inference(
        model=model,
        dataset=dataset,
        beam_size=args.beam_size,
        max_hops=args.max_hops,
        verbose=True,
    )
    
    # Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "checkpoint_dir": str(checkpoint_dir),
            "eval_file": eval_file,
            "beam_size": args.beam_size,
            "max_hops": args.max_hops,
            "inference_temperature": (
                args.inference_temperature
                if args.inference_temperature is not None
                else getattr(config, "inference_temperature", None)
            ),
            "num_samples": len(samples_data),
        },
        "samples": samples_data,
    }
    
    # Save to file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total samples: {len(samples_data)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
