"""SCN Reasoner Model

The main model class implementing SCN (Structurally Constrained Navigation).

SCN uses bidirectional alignment:
- Input side: Soft prompts inject KGE entity vectors into LLM
- Output side: Vector navigation maps LLM hidden states to KGE space

Architecture:
- LLM Backbone (frozen + LoRA)
- Input Projector (KGE → LLM): Maps entity vectors to soft tokens
- Output Projector (LLM → KGE): Maps hidden states to navigation vectors
- Candidate Builder (constructs navigation candidates with fusion)

This solves the "missing entity name" problem by directly injecting entity
semantics as soft tokens, rather than relying on potentially missing text names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Set, Optional, Dict
import re

from .config import SCNConfig
from .llm_backbone import LLMBackbone
from .projector import Projector
from .input_projector import InputProjector
from .candidate_builder import CandidateBuilder
from embeddings.kge_loader import KGELoader
from embeddings.fusion import get_fusion


class SCNReasoner(nn.Module):
    """
    SCN Reasoner: Structurally Constrained Navigation.

    This model performs multi-hop reasoning over knowledge graphs by:
    1. Mapping current entity KGE vector to soft token via InputProjector
    2. Encoding question + path relations + soft token using LLM (+ LoRA)
    3. Projecting the LLM hidden state to KGE space via OutputProjector
    4. Matching the navigation vector with candidate vectors
    5. Selecting the best action (next triple or STOP)

    Entity information is supplied through KGE-derived soft tokens, while
    the text prompt encodes the question and relation history.
    """

    def __init__(
        self,
        config: SCNConfig,
        kge_loader: KGELoader,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize SCN Reasoner.

        Args:
            config: SCN configuration
            kge_loader: Pre-loaded KGE embeddings
            device: Target device
            dtype: Data type for LLM (default: float16)
        """
        super().__init__()

        self.config = config
        self.device = device
        self.dtype = dtype

        # Store KGE loader (frozen, not a module)
        self.kge_loader = kge_loader

        # LLM Backbone with LoRA
        self.llm = LLMBackbone(config, device, dtype)

        # Input Projector: KGE space → LLM space (for soft prompts)
        self.input_projector = InputProjector.from_config(config)
        self.input_projector = self.input_projector.to(device)

        # Output Projector: LLM space → KGE space (for navigation vectors)
        self.output_projector = Projector.from_config(config)
        self.output_projector = self.output_projector.to(device)

        # Candidate Builder (includes learnable v_stop)
        self.candidate_builder = CandidateBuilder(
            kge_loader=kge_loader,
            kge_dim=config.kge_dim,
            fusion_type=config.fusion_type,
            lambda_weight=config.fusion_lambda,
        )
        self.candidate_builder = self.candidate_builder.to(device)

        # Temperature for softmax
        self.temperature = config.temperature

        print(f"SCN Reasoner initialized:")
        print(f"  Fusion type: {config.fusion_type} (λ={config.fusion_lambda})")
        print(f"  Input Projector: KGE({config.kge_dim}) → LLM({config.llm_hidden_size})")
        print(f"  Output Projector: LLM({config.llm_hidden_size}) → KGE({config.kge_dim})")

    def build_text_prompt(
        self,
        question: str,
        path_history: List[Tuple[str, str, str]],
        entity_map: Optional[Dict[str, str]] = None,
        context_text: Optional[str] = None,
    ) -> str:
        """
        Build text prompt from question and path history (without entity names).

        For SCN, entity semantics are injected via soft tokens, not text.
        This prompt only includes the question and relation sequence.

        Format:
            Question: {question}
            Path: {relation_1} -> {relation_2} -> ... -> [ENTITY]
            {context_text}

        Args:
            question: Natural language question
            path_history: List of (head, relation, tail) triples traversed so far
            entity_map: Optional mapping from entity name to ID/placeholder for obfuscation
            context_text: Optional text context to append to the prompt

        Returns:
            Formatted prompt string (entity info comes from soft token)
        """
        # Apply masking to question if map is provided
        processed_question = question
        if entity_map:
            for name, replacement in entity_map.items():
                # Case-insensitive replacement using regex
                processed_question = re.sub(re.escape(name), replacement, processed_question, flags=re.IGNORECASE)
        
        parts = [f"Question: {processed_question}"]

        if path_history:
            # Build path string with only relations
            relations = [r for (h, r, t) in path_history]
            path_str = " -> ".join(relations)
            parts.append(f"Path: {path_str} -> [ENTITY]")
        else:
            parts.append("Path: [START] -> [ENTITY]")
            
        if context_text:
            parts.append(context_text)

        result = "\n".join(parts)
        return result

    def build_prompt(
        self,
        question: str,
        path_history: List[Tuple[str, str, str]],
        entity_map: Optional[Dict[str, str]] = None,
        context_text: Optional[str] = None,
    ) -> str:
        """
        Build input prompt from question and path history.

        Formats entity names and relations for text-only input.

        Format:
            Question: {question}
            Path: {entity_0} -> {relation_1} -> {entity_1} -> ... -> {entity_n}
            {context_text}

        Args:
            question: Natural language question
            path_history: List of (head, relation, tail) triples traversed so far
            entity_map: Optional mapping from entity name to ID/placeholder for obfuscation
            context_text: Optional text context to append to the prompt

        Returns:
            Formatted prompt string
        """
        # Apply masking to question if map is provided
        processed_question = question
        if entity_map:
            # Sort by length descending
            sorted_names = sorted(entity_map.keys(), key=len, reverse=True)
            for name in sorted_names:
                replacement = entity_map[name]
                # Case-insensitive replacement
                processed_question = re.sub(re.escape(name), replacement, processed_question, flags=re.IGNORECASE)

        parts = [f"Question: {processed_question}"]

        if path_history:
            # Build path string with entities and relations
            # Start with first head entity
            start_entity = path_history[0][0]
            if entity_map and start_entity in entity_map:
                start_entity = entity_map[start_entity]
            
            path_str = start_entity
            for h, r, t in path_history:
                tail_entity = t
                if entity_map and tail_entity in entity_map:
                    tail_entity = entity_map[tail_entity]
                path_str += f" -> {r} -> {tail_entity}"
            parts.append(f"Path: {path_str}")
        else:
            parts.append("Path: [START]")
            
        if context_text:
            parts.append(context_text)

        result = "\n".join(parts)
            
        return result

    def get_navigation_vector(
        self,
        question: str,
        path_history: List[Tuple[str, str, str]],
        current_entity: str,
        use_soft_prompt: bool = True,
        entity_map: Optional[Dict[str, str]] = None,
        context_text: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute navigation vector from question, path history, and current entity.

        For SCN (use_soft_prompt=True):
        - Text prompt contains question + relation sequence
        - Current entity is injected as soft token via InputProjector

        For text-only mode (use_soft_prompt=False):
        - Text prompt contains question + full path with entity names
        - No soft token injection

        Args:
            question: Natural language question
            path_history: List of traversed triples
            current_entity: Current entity label
            use_soft_prompt: Whether to use soft prompt injection (default: True)
            entity_map: Optional mapping for entity name obfuscation
            context_text: Optional text to append to prompt

        Returns:
            Navigation vector v_nav of shape [kge_dim], L2-normalized
        """
        if use_soft_prompt:
            # SCN mode: use hybrid input with soft token
            # Note: SCN's text prompt (build_text_prompt) typically doesn't include entity names 
            # in the path anyway, only relations. But it does include the Question.
            # So entity_map is relevant for obfuscating the Question.
            text_prompt = self.build_text_prompt(question, path_history, entity_map, context_text)

            # Get entity KGE vector and project to LLM space
            entity_kge_vec = self.kge_loader.get_entity_embedding(current_entity)
            entity_soft_token = self.input_projector(entity_kge_vec)

            # Encode hybrid input (text + soft token)
            h_last = self.llm.encode_hybrid(text_prompt, entity_soft_token)
        else:
            # Text-only prompt
            # Apply the entity mapping to names in the text path.
            prompt = self.build_prompt(question, path_history, entity_map, context_text)
            h_last = self.llm.encode_text(prompt)

        # Convert to float32 for projector if needed
        h_last = h_last.float()

        # Project to KGE space
        v_nav = self.output_projector(h_last)

        return v_nav

    def compute_action_logits(
        self,
        v_nav: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute action logits via cosine similarity.

        Args:
            v_nav: Navigation vector, shape [kge_dim]
            candidate_matrix: Candidate vectors, shape [num_candidates, kge_dim]
                              Row 0 is STOP, rows 1..N are neighbor actions

        Returns:
            Logits for each action, shape [num_candidates]
        """
        # Both v_nav and candidate_matrix should be L2-normalized
        # Cosine similarity = dot product for unit vectors
        similarities = torch.matmul(candidate_matrix, v_nav)

        # Scale by temperature
        logits = similarities / self.temperature

        return logits

    def compute_action_probs(
        self,
        v_nav: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute action probabilities.

        Args:
            v_nav: Navigation vector, shape [kge_dim]
            candidate_matrix: Candidate vectors, shape [num_candidates, kge_dim]

        Returns:
            Probabilities for each action, shape [num_candidates]
        """
        logits = self.compute_action_logits(v_nav, candidate_matrix)
        return F.softmax(logits, dim=-1)

    def compute_action_log_probs(
        self,
        v_nav: torch.Tensor,
        candidate_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute action log-probabilities.

        Args:
            v_nav: Navigation vector, shape [kge_dim]
            candidate_matrix: Candidate vectors, shape [num_candidates, kge_dim]

        Returns:
            Log-probabilities for each action, shape [num_candidates]
        """
        logits = self.compute_action_logits(v_nav, candidate_matrix)
        return F.log_softmax(logits, dim=-1)

    def forward(
        self,
        question: str,
        path_history: List[Tuple[str, str, str]],
        current_entity: str,
        subgraph: Set[Tuple[str, str, str]],
        visited: Set[str],
        use_soft_prompt: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str, str]]]:
        """
        Forward pass: compute action probabilities for current state.

        Args:
            question: Natural language question
            path_history: List of traversed triples
            current_entity: Current entity label
            subgraph: Set of available triples
            visited: Set of visited entity labels
            use_soft_prompt: Whether to use soft prompt injection (default: True)

        Returns:
            Tuple of:
            - Action log-probabilities, shape [num_candidates]
            - Navigation vector v_nav, shape [kge_dim]
            - List of candidate triples (index 0 = STOP)
        """
        # Get navigation vector
        v_nav = self.get_navigation_vector(
            question=question,
            path_history=path_history,
            current_entity=current_entity,
            use_soft_prompt=use_soft_prompt,
        )

        # Build candidate matrix
        candidate_matrix, neighbors = self.candidate_builder.build_candidates_for_entity(
            current_entity=current_entity,
            subgraph=subgraph,
            visited=visited,
        )

        # Compute log-probabilities
        log_probs = self.compute_action_log_probs(v_nav, candidate_matrix)

        return log_probs, v_nav, neighbors

    def select_action_greedy(
        self,
        question: str,
        path_history: List[Tuple[str, str, str]],
        current_entity: str,
        subgraph: Set[Tuple[str, str, str]],
        visited: Set[str],
        use_soft_prompt: bool = True,
    ) -> Tuple[Optional[Tuple[str, str, str]], float, bool]:
        """
        Greedy action selection: choose the highest probability action.

        Args:
            question: Natural language question
            path_history: List of traversed triples
            current_entity: Current entity label
            subgraph: Set of available triples
            visited: Set of visited entity labels
            use_soft_prompt: Whether to use soft prompt injection (default: True)

        Returns:
            Tuple of:
            - Selected triple (or None if STOP)
            - Log-probability of selected action
            - Boolean indicating if STOP was selected
        """
        log_probs, _, neighbors = self.forward(
            question=question,
            path_history=path_history,
            current_entity=current_entity,
            subgraph=subgraph,
            visited=visited,
            use_soft_prompt=use_soft_prompt,
        )

        # Greedy selection
        best_idx = torch.argmax(log_probs).item()
        best_log_prob = log_probs[best_idx].item()

        if best_idx == 0:
            # STOP selected
            return None, best_log_prob, True
        else:
            # Return the selected triple
            selected_triple = neighbors[best_idx - 1]
            return selected_triple, best_log_prob, False

    def get_trainable_parameters(self):
        """
        Get all trainable parameters.

        Returns iterator over:
        - LoRA parameters (if enabled)
        - Input Projector parameters
        - Output Projector parameters
        - v_stop parameter
        """
        # LoRA parameters
        if self.config.use_lora:
            for param in self.llm.get_trainable_parameters():
                yield param

        # Input Projector parameters
        for param in self.input_projector.parameters():
            yield param

        # Output Projector parameters
        for param in self.output_projector.parameters():
            yield param

        # Candidate builder parameters (v_stop)
        for param in self.candidate_builder.parameters():
            yield param

    def get_parameter_groups(self) -> List[dict]:
        """
        Get parameter groups with different learning rates.

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []

        # LoRA parameters
        if self.config.use_lora:
            lora_params = list(self.llm.get_trainable_parameters())
            if lora_params:
                groups.append({
                    "params": lora_params,
                    "lr": self.config.lora_lr,
                    "name": "lora",
                })

        # Input Projector parameters (SCN)
        input_proj_params = list(self.input_projector.parameters())
        if input_proj_params:
            groups.append({
                "params": input_proj_params,
                "lr": self.config.input_projector_lr,
                "name": "input_projector",
            })

        # Output Projector parameters
        output_proj_params = list(self.output_projector.parameters())
        if output_proj_params:
            groups.append({
                "params": output_proj_params,
                "lr": self.config.projector_lr,
                "name": "output_projector",
            })

        # Candidate builder parameters (v_stop)
        candidate_params = list(self.candidate_builder.parameters())
        if candidate_params:
            groups.append({
                "params": candidate_params,
                "lr": self.config.learning_rate,
                "name": "candidate_builder",
            })

        return groups

    def count_parameters(self) -> dict:
        """
        Count parameters by component.

        Returns:
            Dict with parameter counts
        """
        counts = {
            "lora": 0,
            "input_projector": 0,
            "output_projector": 0,
            "candidate_builder": 0,
            "total_trainable": 0,
        }

        if self.config.use_lora:
            # 统计LoRA参数（通过参数名判断，不依赖requires_grad）
            counts["lora"] = sum(
                p.numel() for name, p in self.llm.model.named_parameters()
                if "lora_" in name
            )

        counts["input_projector"] = sum(p.numel() for p in self.input_projector.parameters())
        counts["output_projector"] = sum(p.numel() for p in self.output_projector.parameters())
        counts["candidate_builder"] = sum(p.numel() for p in self.candidate_builder.parameters())
        counts["total_trainable"] = sum(
            p.numel() for p in self.get_trainable_parameters()
        )

        return counts