"""
SCN (Structurally Constrained Navigation)

Knowledge Graph Question Answering with bidirectional alignment:
- Input side: Soft prompts inject KGE entity vectors into LLM
- Output side: Vector navigation maps LLM hidden states to KGE space

Core modules for vector-space reasoning over knowledge graphs.

Note: LLMBackbone and SCNReasoner require transformers/peft libraries.
Import them explicitly when needed to avoid dependency issues.
"""

from .config import SCNConfig
from .projector import Projector
from .input_projector import InputProjector
from .losses import InfoNCELoss, StepwiseInfoNCELoss, ActionLoss
from .state import ReasoningState, StateManager

# Lazy imports for heavy dependencies
def get_llm_backbone():
    from .llm_backbone import LLMBackbone
    return LLMBackbone

def get_candidate_builder():
    from .candidate_builder import CandidateBuilder
    return CandidateBuilder

def get_scn_reasoner():
    from .model import SCNReasoner
    return SCNReasoner

def get_beam_search():
    from .inference import BeamSearchInference, InferenceResult
    return BeamSearchInference, InferenceResult

def get_input_projector():
    from .input_projector import InputProjector
    return InputProjector

__all__ = [
    "SCNConfig",
    "Projector",
    "InputProjector",
    "InfoNCELoss",
    "StepwiseInfoNCELoss",
    "ActionLoss",
    "ReasoningState",
    "StateManager",
    "get_llm_backbone",
    "get_candidate_builder",
    "get_scn_reasoner",
    "get_beam_search",
    "get_input_projector",
]
