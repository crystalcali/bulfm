from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]
READONLY_HINT = 'readonly=True forbids fill/type, NOT clicking. A readonly date textbox can itself be clicked\nto open its calendar. Click its bid, then inspect the NEW page and use the calendar controls.'
SYSTEM_PROMPT = """You control a browser. Complete the goal using the current page structure.
Reply with exactly ONE Python-style action and no explanation or markdown.
Actions: click('bid'), fill('bid', 'text'), press('bid', 'Enter'),
select_option('bid', 'value'), scroll(0, 500), noop().
The number in brackets is the element's bid, e.g. [13] button 'OK' => click('13').
Use only IDs present in the page. Observe errors and correct your next action.
After an error, do not assume the action succeeded.""" + "\n" + READONLY_HINT


class CurriculumTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str = Field(pattern=r"^miniwob\.[a-z0-9-]+$")
    episode_steps: int = Field(default=8, ge=1, le=100)


class CurriculumStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    tasks: list[CurriculumTask] = Field(min_length=1)


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stages: list[CurriculumStage] = Field(min_length=1)
    replay_fraction: float = Field(default=0.3, ge=0, lt=1)
    window_episodes: int = Field(default=32, ge=2)
    min_episodes: int = Field(default=16, ge=2)
    promote_success: float = Field(default=0.6, gt=0, le=1)
    promote_valid: float = Field(default=0.8, ge=0, le=1)
    eval_episodes_per_task: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def consistent(self):
        names = [t.task for stage in self.stages for t in stage.tasks]
        if len(names) != len(set(names)):
            raise ValueError("Curriculum task names must be unique")
        if self.min_episodes > self.window_episodes:
            raise ValueError("min_episodes must not exceed window_episodes")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "LiquidAI/LFM3-1.7B"
    model_revision: str = "main"
    trust_remote_code: bool = True
    task: str = "miniwob.click-test"
    curriculum: CurriculumConfig | None = None
    seed: int = Field(default=23, ge=0, le=1_000_000_000)
    eval_seed: int = Field(default=1_000_000_001, ge=1_000_000_001, le=2_000_000_000)
    steps: int = Field(default=100, ge=1, le=100_000)
    group_size: int = Field(default=4, ge=2, le=32)
    groups_per_step: int = Field(default=1, ge=1, le=32)
    num_envs: int = Field(default=1, ge=1, le=16)
    generation_batch_size: int = Field(default=4, ge=1, le=16)
    generation_use_cache: bool = False
    env_timeout_seconds: float = Field(default=60, ge=5, le=600)
    episode_steps: int = Field(default=6, ge=1, le=100)
    context_tokens: int = Field(default=2048, ge=256)
    completion_tokens: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0)
    warmup_steps: int = Field(default=5, ge=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    beta: float = Field(default=0.04, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_targets: list[str] = Field(default_factory=lambda: LORA_TARGETS.copy())
    gradient_checkpointing: bool = True
    precision: Literal["bf16", "fp32"] = "bf16"
    device: Literal["cuda", "cpu"] = "cuda"
    save_every: int = Field(default=10, ge=1)
    keep_checkpoints: int = Field(default=2, ge=1)
    eval_episodes: int = Field(default=20, ge=1)
    eval_every: int = Field(default=25, ge=0)
    baseline_eval: bool = True
    system_prompt: str = SYSTEM_PROMPT
    miniwob_url: str | None = None
    headless: bool = True
    timeout_ms: int = Field(default=5000, ge=1000)

    @model_validator(mode="after")
    def consistent(self):
        if self.completion_tokens + 64 >= self.context_tokens:
            raise ValueError("context_tokens must leave at least 64 tokens for the prompt")
        if self.beta and not self.lora:
            raise ValueError("Full tuning requires beta: 0 (no second reference model allocated)")
        if self.seed + self.steps * self.groups_per_step >= self.eval_seed:
            raise ValueError("Training and evaluation seed ranges overlap")
        return self

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def save(self, path: str | Path):
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
