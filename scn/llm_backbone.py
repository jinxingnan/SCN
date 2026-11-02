"""
LLM Backbone with LoRA

Provides a frozen LLM backbone with optional LoRA adapters for fine-tuning.
Extracts hidden states from the last token position for downstream projection.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, List, Tuple

from .config import SCNConfig


class LLMBackbone(nn.Module):
    """
    LLM backbone with frozen base weights and optional LoRA adapters.

    This module:
    1. Loads a pretrained causal LLM (e.g., Llama-2)
    2. Freezes all base model parameters
    3. Optionally injects LoRA adapters for fine-tuning
    4. Extracts hidden states from the last token position

    The hidden states are used by the Projector to generate navigation vectors.
    """

    def __init__(
        self,
        config: SCNConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize LLM backbone.

        Args:
            config: SCN configuration
            device: Target device for the model
            dtype: Data type for model weights (default: float16 for memory efficiency)
        """
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )

        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply LoRA if enabled
        if config.use_lora:
            self._apply_lora()

        # Enable gradient checkpointing if requested (must be after LoRA)
        if config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Store hidden size
        self.hidden_size = self.model.config.hidden_size

        # Verify hidden size matches config
        if self.hidden_size != config.llm_hidden_size:
            raise ValueError(
                f"LLM hidden size ({self.hidden_size}) does not match "
                f"config.llm_hidden_size ({config.llm_hidden_size})"
            )

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"LoRA applied to: {self.config.lora_target_modules}")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def _enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing to reduce memory usage.

        Gradient checkpointing trades compute for memory by not storing
        intermediate activations during the forward pass. Instead, they
        are recomputed during the backward pass.

        Benefits:
        - Saves 30-50% VRAM (critical for large models like Llama-2-7B)
        - Allows training with larger batch sizes or longer sequences

        Drawbacks:
        - Training is ~20% slower due to recomputation
        """
        # Enable input gradients (required for gradient checkpointing with frozen embeddings)
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()

        # Enable gradient checkpointing on the base model
        # For PEFT models, we need to access the base model
        if hasattr(self.model, 'base_model'):
            # PEFT model structure: PeftModel -> base_model (LoraModel) -> model (original)
            if hasattr(self.model.base_model, 'model'):
                base_model = self.model.base_model.model
            else:
                base_model = self.model.base_model
        else:
            # Regular model (no PEFT)
            base_model = self.model

        # Enable gradient checkpointing
        if hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (saves ~30-50% VRAM, ~20% slower)")
        else:
            print("Warning: gradient_checkpointing_enable not available on this model")

    def tokenize(
        self,
        text: str,
        max_length: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize input text.

        Args:
            text: Input text string
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask), both of shape [1, seq_len]
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        return input_ids, attention_mask

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts with padding.

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length

        Returns:
            Tuple of (input_ids, attention_mask), both of shape [batch, seq_len]
        """
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        return input_ids, attention_mask

    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get hidden state at the last token position.

        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            attention_mask: Attention mask, shape [batch, seq_len]

        Returns:
            Hidden state at last token, shape [batch, hidden_size]
        """
        # Forward pass with hidden states output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        # Get last layer hidden states: [batch, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Find the position of the last non-padding token for each sequence
        # attention_mask: 1 for real tokens, 0 for padding
        # Sum along seq dimension gives the sequence length
        seq_lengths = attention_mask.sum(dim=1)  # [batch]

        # Get the hidden state at the last real token position
        batch_size = input_ids.shape[0]
        last_hidden = torch.stack([
            hidden_states[i, seq_lengths[i] - 1, :]
            for i in range(batch_size)
        ], dim=0)  # [batch, hidden_size]

        return last_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: get last token hidden state.

        This is the main interface for the SCN model.

        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            attention_mask: Attention mask, shape [batch, seq_len]

        Returns:
            Hidden state at last token, shape [batch, hidden_size]
        """
        return self.get_last_hidden_state(input_ids, attention_mask)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Convenience method: tokenize and get hidden state for a single text.

        Args:
            text: Input text string

        Returns:
            Hidden state, shape [hidden_size]
        """
        input_ids, attention_mask = self.tokenize(text)
        hidden = self.get_last_hidden_state(input_ids, attention_mask)
        return hidden.squeeze(0)  # Remove batch dimension

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Convenience method: tokenize and get hidden states for multiple texts.

        Args:
            texts: List of input text strings

        Returns:
            Hidden states, shape [batch, hidden_size]
        """
        input_ids, attention_mask = self.tokenize_batch(texts)
        return self.get_last_hidden_state(input_ids, attention_mask)

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get word embeddings for token IDs.

        Args:
            input_ids: Token IDs, shape [batch, seq_len] or [seq_len]

        Returns:
            Word embeddings, shape [batch, seq_len, hidden_size] or [seq_len, hidden_size]
        """
        # Get embedding layer from model
        # Handle both regular models and PEFT models
        if hasattr(self.model, 'get_input_embeddings'):
            embed_layer = self.model.get_input_embeddings()
        elif hasattr(self.model, 'base_model'):
            embed_layer = self.model.base_model.get_input_embeddings()
        else:
            embed_layer = self.model.model.embed_tokens

        return embed_layer(input_ids)

    def encode_hybrid(
        self,
        text: str,
        entity_soft_token: Optional[torch.Tensor] = None,
        max_length: int = 2048,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Encode hybrid input: text + optional entity soft token.

        This is the core method for SCN architecture.
        The entity soft token is appended to the text embeddings,
        allowing the LLM to process both natural language and
        entity semantics in a unified manner.

        Args:
            text: Natural language text (question + path relations)
            entity_soft_token: Entity vector projected to LLM space
                              Shape: [hidden_size]
                              If None, behaves like encode_text()
            max_length: Maximum sequence length for text
            output_attentions: If True, return attention weights

        Returns:
            If output_attentions=False:
                Hidden state at the last position (entity token if provided)
                Shape: [hidden_size]
            If output_attentions=True:
                Tuple (hidden_state, attentions)
        """
        # Tokenize text
        input_ids, attention_mask = self.tokenize(text, max_length=max_length)
        # input_ids: [1, seq_len], attention_mask: [1, seq_len]

        # Get text embeddings
        text_embeds = self.get_text_embeddings(input_ids)
        # text_embeds: [1, seq_len, hidden_size]

        # If no entity soft token, use standard encoding
        if entity_soft_token is None:
            outputs = self.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
            
            # Return last token's hidden state
            last_hidden = hidden_states[0, -1, :]  # [hidden_size]
            
            if output_attentions:
                return last_hidden, outputs.attentions
            return last_hidden

        # Append entity soft token to embeddings
        # entity_soft_token: [hidden_size]
        entity_token = entity_soft_token.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        entity_token = entity_token.to(text_embeds.dtype)  # Match dtype

        # Concatenate: [text_embeds, entity_token]
        inputs_embeds = torch.cat([text_embeds, entity_token], dim=1)
        # inputs_embeds: [1, seq_len + 1, hidden_size]

        # Extend attention mask for the entity token
        entity_attn = torch.ones(1, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([attention_mask, entity_attn], dim=1)
        # extended_attention_mask: [1, seq_len + 1]

        # Forward pass through LLM
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [1, seq_len + 1, hidden_size]

        # Return the hidden state at the last position (entity token position)
        # This captures the LLM's understanding of both the question and the current entity
        last_hidden = hidden_states[0, -1, :]  # [hidden_size]

        if output_attentions:
            return last_hidden, outputs.attentions
            
        return last_hidden

    def encode_hybrid_batch(
        self,
        texts: List[str],
        entity_soft_tokens: Optional[torch.Tensor] = None,
        max_length: int = 2048,
    ) -> torch.Tensor:
        """
        Batch version of encode_hybrid.

        Args:
            texts: List of text strings
            entity_soft_tokens: Entity vectors projected to LLM space
                               Shape: [batch, hidden_size]
                               If None, behaves like encode_texts()
            max_length: Maximum sequence length for text

        Returns:
            Hidden states at last positions
            Shape: [batch, hidden_size]
        """
        batch_size = len(texts)

        # Tokenize with padding
        input_ids, attention_mask = self.tokenize_batch(texts, max_length=max_length)
        # input_ids: [batch, max_seq_len], attention_mask: [batch, max_seq_len]

        # Get text embeddings
        text_embeds = self.get_text_embeddings(input_ids)
        # text_embeds: [batch, max_seq_len, hidden_size]

        # If no entity soft tokens, use standard encoding
        if entity_soft_tokens is None:
            outputs = self.model(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]

            # Get last real token position for each sequence
            seq_lengths = attention_mask.sum(dim=1)  # [batch]
            last_hidden = torch.stack([
                hidden_states[i, seq_lengths[i] - 1, :]
                for i in range(batch_size)
            ], dim=0)
            return last_hidden

        # Append entity soft tokens
        # entity_soft_tokens: [batch, hidden_size]
        entity_tokens = entity_soft_tokens.unsqueeze(1)  # [batch, 1, hidden_size]
        entity_tokens = entity_tokens.to(text_embeds.dtype)

        # Concatenate: [text_embeds, entity_tokens]
        inputs_embeds = torch.cat([text_embeds, entity_tokens], dim=1)
        # inputs_embeds: [batch, max_seq_len + 1, hidden_size]

        # Extend attention mask
        entity_attn = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        extended_attention_mask = torch.cat([attention_mask, entity_attn], dim=1)

        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # [batch, max_seq_len + 1, hidden_size]

        # Return the hidden state at the last position (entity token position)
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]

        return last_hidden

    def get_trainable_parameters(self):
        """Get iterator over trainable parameters (LoRA parameters)."""
        return (p for p in self.model.parameters() if p.requires_grad)

    def get_num_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_lora_weights(self, path: str) -> None:
        """Save LoRA adapter weights."""
        if not self.config.use_lora:
            raise ValueError("LoRA is not enabled, nothing to save")
        self.model.save_pretrained(path)

    def load_lora_weights(self, path: str) -> None:
        """
        Load LoRA adapter weights from a saved checkpoint.

        Note: This replaces the current LoRA adapter with the saved one.
        The base model is extracted and a new PeftModel is created.
        """
        if not self.config.use_lora:
            raise ValueError("LoRA is not enabled, nothing to load")

        from peft import PeftModel

        # Get the original base model (not the LoraModel wrapper)
        # self.model is PeftModel -> self.model.base_model is LoraModel
        # -> self.model.base_model.model is the original HuggingFace model
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            base_model = self.model.base_model.model
        else:
            # Fallback: model might not be wrapped yet
            base_model = self.model

        # Create new PeftModel from saved adapter
        self.model = PeftModel.from_pretrained(base_model, path)
