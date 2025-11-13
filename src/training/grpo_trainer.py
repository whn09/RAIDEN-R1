"""
GRPO (Group Relative Policy Optimization) Trainer for RAIDEN-R1

Implements the training loop using GRPO with VRAR rewards
"""

import torch
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from trl import AutoModelForCausalLMWithValueHead
from accelerate import Accelerator

import sys
sys.path.append(str(Path(__file__).parent.parent))

from rewards.vrar import VRARCalculator, RoleAwarenessValidator
from data.collection import RolePlayingSample


@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    learning_rate: float = 3e-6
    batch_size: int = 4
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples_per_prompt: int = 4  # For GRPO group sampling
    kl_penalty: float = 0.1
    accuracy_weight: float = 0.7
    format_weight: float = 0.3
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./outputs"
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50
    warmup_steps: int = 100


class GRPOTrainer:
    """
    GRPO Trainer for role-aware language models

    Implements Group Relative Policy Optimization with VRAR
    """

    def __init__(
        self,
        config: GRPOConfig,
        train_dataset: List[RolePlayingSample],
        eval_dataset: Optional[List[RolePlayingSample]] = None
    ):
        """
        Args:
            config: GRPO training configuration
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="bf16" if config.use_bf16 else "no"
        )

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # Initialize reward calculator
        self.reward_calculator = VRARCalculator(
            accuracy_weight=config.accuracy_weight,
            format_weight=config.format_weight
        )
        self.role_validator = RoleAwarenessValidator()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Prepare for distributed training
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # Statistics
        self.global_step = 0
        self.epoch = 0

    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.config.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer

    def train(self):
        """Main training loop"""
        print("Starting GRPO training...")
        print(f"Total samples: {len(self.train_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.num_epochs}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            self._train_epoch()

            if self.eval_dataset:
                self._evaluate()

            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}")

        print("\nTraining completed!")

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        # Create batches
        num_batches = len(self.train_dataset) // self.config.batch_size

        progress_bar = tqdm(
            range(num_batches),
            desc=f"Training Epoch {self.epoch + 1}"
        )

        for batch_idx in progress_bar:
            batch = self._get_batch(batch_idx)

            loss, metrics = self._train_step(batch)

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_reward": f"{metrics.get('avg_reward', 0):.4f}"
            })

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics(loss, metrics)

            self.global_step += 1

    def _get_batch(self, batch_idx: int) -> List[RolePlayingSample]:
        """Get a batch of samples"""
        start_idx = batch_idx * self.config.batch_size
        end_idx = start_idx + self.config.batch_size
        return self.train_dataset[start_idx:end_idx]

    def _train_step(self, batch: List[RolePlayingSample]) -> tuple:
        """
        Single training step with GRPO

        GRPO process:
        1. Generate multiple responses per prompt (group)
        2. Calculate rewards for each response
        3. Compute relative advantages within group
        4. Update policy based on relative performance
        """
        total_loss = 0.0
        all_rewards = []

        for sample in batch:
            # Generate multiple responses for this prompt (GRPO group sampling)
            responses = self._generate_group_responses(sample)

            # Calculate rewards for all responses
            rewards = self._calculate_rewards(sample, responses)
            all_rewards.extend(rewards)

            # Calculate relative advantages
            advantages = self._calculate_relative_advantages(rewards)

            # Compute policy loss
            loss = self._compute_grpo_loss(sample, responses, advantages)

            total_loss += loss.item()

        # Backward pass
        avg_loss = total_loss / len(batch)
        self.accelerator.backward(torch.tensor(avg_loss, requires_grad=True))

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        metrics = {
            "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "num_samples": len(batch) * self.config.num_samples_per_prompt
        }

        return avg_loss, metrics

    def _generate_group_responses(
        self,
        sample: RolePlayingSample
    ) -> List[str]:
        """
        Generate multiple responses for a single prompt (GRPO group)

        Args:
            sample: Input sample

        Returns:
            List of generated responses
        """
        prompt = self._format_prompt(sample)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.model.device)

        responses = []

        # Generate multiple responses with sampling
        for _ in range(self.config.num_samples_per_prompt):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def _format_prompt(self, sample: RolePlayingSample) -> str:
        """Format a sample into a prompt for the model"""
        profile_str = json.dumps(sample.character_profile, indent=2)

        prompt = f"""You are roleplaying as {sample.character}.

Character Profile:
{profile_str}

User Query: {sample.query}

Please provide a detailed response with step-by-step reasoning, staying in character.

Response:"""

        return prompt

    def _calculate_rewards(
        self,
        sample: RolePlayingSample,
        responses: List[str]
    ) -> List[float]:
        """Calculate VRAR rewards for all responses"""
        rewards = []

        for response in responses:
            reward_result = self.reward_calculator.calculate_reward(
                response=response,
                ground_truth=sample.ground_truth,
                validation_type=sample.validation_type
            )
            rewards.append(reward_result.total_reward)

        return rewards

    def _calculate_relative_advantages(self, rewards: List[float]) -> List[float]:
        """
        Calculate relative advantages within the group (GRPO)

        Advantage = (reward - mean_reward) / (std_reward + 1e-8)
        """
        if len(rewards) <= 1:
            return [0.0] * len(rewards)

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5

        advantages = [
            (r - mean_reward) / (std_reward + 1e-8)
            for r in rewards
        ]

        return advantages

    def _compute_grpo_loss(
        self,
        sample: RolePlayingSample,
        responses: List[str],
        advantages: List[float]
    ) -> torch.Tensor:
        """
        Compute GRPO loss

        Loss = -mean(advantage * log_prob) + kl_penalty
        """
        prompt = self._format_prompt(sample)

        total_loss = torch.tensor(0.0, device=self.model.device)

        for response, advantage in zip(responses, advantages):
            # Tokenize prompt + response
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.model.device)

            # Get log probabilities
            with torch.enable_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                log_probs = -outputs.loss

            # Policy gradient loss
            policy_loss = -advantage * log_probs

            total_loss += policy_loss

        # Average over group
        avg_loss = total_loss / len(responses)

        return avg_loss

    def _evaluate(self):
        """Evaluate on eval dataset"""
        if not self.eval_dataset:
            return

        print("\nRunning evaluation...")
        self.model.eval()

        total_reward = 0.0
        sbk_scores = []
        cm_scores = []

        with torch.no_grad():
            for sample in tqdm(self.eval_dataset[:100], desc="Evaluating"):  # Limit for speed
                # Generate response
                responses = self._generate_group_responses(sample)
                best_response = responses[0]  # Use first sample

                # Calculate reward
                reward_result = self.reward_calculator.calculate_reward(
                    response=best_response,
                    ground_truth=sample.ground_truth,
                    validation_type=sample.validation_type
                )
                total_reward += reward_result.total_reward

                # Calculate SBK and CM
                sbk = self.role_validator.validate_script_knowledge(
                    best_response,
                    sample.character_profile
                )
                sbk_scores.append(sbk)

        avg_reward = total_reward / min(len(self.eval_dataset), 100)
        avg_sbk = sum(sbk_scores) / len(sbk_scores) if sbk_scores else 0

        print(f"Eval Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average SBK: {avg_sbk:.4f}")

        self.model.train()

    def _log_metrics(self, loss: float, metrics: Dict[str, Any]):
        """Log training metrics"""
        log_data = {
            "step": self.global_step,
            "epoch": self.epoch,
            "loss": loss,
            **metrics
        }

        # Print to console
        if self.global_step % (self.config.logging_steps * 10) == 0:
            print(f"\nStep {self.global_step}: Loss={loss:.4f}, " +
                  f"Reward={metrics.get('avg_reward', 0):.4f}")

        # Save to file
        log_file = Path(self.config.output_dir) / "training_log.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint"""
        output_dir = Path(self.config.output_dir) / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint to {output_dir}")

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        config_file = output_dir / "grpo_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

        print(f"Checkpoint saved successfully")
