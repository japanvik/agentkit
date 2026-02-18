# Next Tasks

- [ ] Implement plan-oriented multi-agent orchestration in Sophia/ToolBrain.
- [x] Generate per-message action items in planner state (with dependencies + completion criteria) using an LLM-backed planner pass.
- [x] Support a dedicated planner model in config (`planner_model`), defaulting to `model` when unset.
- [ ] Drive execution from planner action state instead of only brain/tool loop heuristics.
- [ ] Enforce non-terminal send loop prevention at planner level (single final response once completion criteria are met).
- [ ] Add integration tests for end-to-end multi-agent task completion and final-response gating.
