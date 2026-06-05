# ADR-0002: Reusability foundation (contract + ToyFPS) before MCP/RAG

**Status**: accepted (2026-06-03)

## Context

Portability is a primary line of the project. MCP (standardising the game
interface as a server) and RAG-B (knowledge-driven test generation) are
attractive, but the adapter contract was not yet explicit and there was only one
game implementation (ViZDoom).

## Decision

Sequence the reusability ladder: (1) make the adapter contract explicit +
(2) a second reference implementation (ToyFPS) FIRST; (3) MCP and (4) RAG-B come
after. The contract is emerged from the second implementation, not designed up
front.

## Consequences

- MCP is rung 3, not step 1: it protocol-ises a contract that must first be clear.
  Standardising an unclear interface just standardises the mess.
- RAG-B is the crown (rung 4) and needs the contract + a richer action library.
- LLM-assisted adapter generation follows the same lesson: a prompt template
  (prove the flow) before an `adapter-generator` skill (package it). See
  `Doc/v2-roadmap.md` and `Doc/adapter-generation-prompt.md`.
