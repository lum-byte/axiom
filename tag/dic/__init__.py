"""DirectlyInjectContext (DIC) package for TAG context fusion."""

from tag.dic.assembler import DirectlyInjectContextAssembler
from tag.dic.gbnf_dsl import QueryExpansionEngine, QueryExpansionResult
from tag.dic.hybrid_search import HybridFusionRanker
from tag.dic.injector import DirectContextInjector
from tag.dic.mcp_process import ExternalMCPAnchorClient

__all__ = [
    "DirectContextInjector",
    "DirectlyInjectContextAssembler",
    "ExternalMCPAnchorClient",
    "HybridFusionRanker",
    "QueryExpansionEngine",
    "QueryExpansionResult",
]
