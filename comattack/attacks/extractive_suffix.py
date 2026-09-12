import math
import string
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification)

from .coma_utils import (
    AttackConfig,
    NPEncoder,
    get_embedding_layer,
    get_embedding_matrix,
    get_token_embeddings,
    get_nonascii_toks_stable,
    sample_control,
    find_slices_from_token,
    find_suffix_from_token,
    _ppl_control_loss,
    _token_cls_loss,
)
from ..spc_query_suffix import configure_right_padding, target_span_mask, validate_prompt_batch

class AttackEvaluator(object):
    """
    A class to evaluate the attack
    """
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


class AttackforLLMLingua1(object):
    """
    A class to attack the llmlingua1
    """
    def __init__(self, config: Optional[AttackConfig] = None, **config_kwargs):
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = AttackConfig(**config_kwargs)
        else:
            raise ValueError("AttackforLLMLingua1 requires config=AttackConfig(model_name='...') or model_name='...'")
        self.model_name = self.config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, return_offset_mapping=True)
        self.nonascii_toks = get_nonascii_toks_stable(self.tokenizer)

        self.device = self.model.device
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.config.seed)
        self.vocab_size = get_embedding_matrix(self.model).shape[0]
        self.best_loss = float('inf')
        self.best_candidates = None

    def llmlingua1_loss(self, logits, suffix_slice, target_slice, ids):
        """
        The loss function for the llmlingua1
        """
        return -1.0 * _ppl_control_loss(logits, suffix_slice, target_slice, ids)


    def llmlingua1_gradients(
        self,
        model: AutoModelForCausalLM,
        full_ids: torch.Tensor,
        suffix_slice: slice,
        target_slice: slice,
        device: torch.device,
    ):
        """
        Compute gradients of the LLMLingua1 loss w.r.t. one-hot suffix representation.
        Returns gradient tensor of shape [suffix_len, vocab_size].
        """
        full_ids = full_ids.squeeze(0) # shape = [T]
        suffix_ids = full_ids[suffix_slice.start:suffix_slice.stop].to(device=device, dtype=torch.long)
        suffix_len = suffix_ids.numel()
        W = get_embedding_matrix(model) # shape = [vocab, dim]
        vocab_size = self.vocab_size
        base_embeds = get_token_embeddings(model, full_ids.unsqueeze(0)).detach() # shape = [1, T, dim]

        one_hot = torch.zeros((suffix_len, vocab_size), device=device, dtype=torch.float32)
        one_hot.scatter_(1, suffix_ids.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_embeds = (one_hot.float() @ W.float()).to(dtype=W.dtype).unsqueeze(0) # [1, suffix_len, dim]
        full_embeds = torch.cat([
            base_embeds[:, :suffix_slice.start, :],
            suffix_embeds,
            base_embeds[:, suffix_slice.stop:, :],
        ], dim=1)

        model.zero_grad(set_to_none=True)
        output = model(inputs_embeds=full_embeds, input_ids=None)
        logits = output["logits"]
        loss = self.llmlingua1_loss(logits, suffix_slice, target_slice, full_ids)
        loss.backward()

        return one_hot.grad.squeeze(0) # shape = [suffix_len, vocab_size]


    def _sample_control(self, control_toks, grad, sample_batch_size, topk=256, sample_temp=1.0, generator=None):
        """Delegate to standalone sample_control."""
        return sample_control(
            control_toks=control_toks, grad=grad,
            nonascii_toks=self.nonascii_toks, vocab_size=self.vocab_size,
            sample_batch_size=sample_batch_size, topk=topk,
            sample_temp=sample_temp, generator=generator,
        )

    def evaluate_candidates(self, eval_logits, eval_full_ids, suffix_slice, target_slice):
        """
        This function is used to evaluate the candidates
        """
        eval_batch_size = self.config.eval_batch_size
        batch_num = math.ceil(eval_full_ids.size(0) / eval_batch_size)
        losses = []
        with torch.no_grad():
            for i in range(batch_num):
                batch_full_ids = eval_full_ids[i*eval_batch_size: (i+1)*eval_batch_size, :]
                batch_eval_logits = eval_logits[i*eval_batch_size: (i+1)*eval_batch_size, :, :]
                batch_loss = self.llmlingua1_loss(
                    logits=batch_eval_logits,
                    suffix_slice=suffix_slice,
                    target_slice=target_slice,
                    ids=batch_full_ids,
                )
                losses.append(batch_loss.detach().cpu())
        # shape of losses list is [batch_num, eval_batch_size]
        losses = torch.cat(losses, dim=0).numpy() # shape = [sample_batch_size = batch_num * eval_batch_size]
        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])
        return min_loss, min_idx

    def attack(self, prompt: str, guardrail_sentence: str, guardrail_keyword: str):
        """
        This function is used to attack the llmlingua1
        """
        device = self.model.device
        suffix_slice, target_slice = find_slices_from_token(self.tokenizer, prompt, guardrail_sentence, guardrail_keyword, self.config.suffix_length)
        full_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape = [1, T]
        if self.best_candidates is not None:
            full_ids[:, suffix_slice] = self.best_candidates.to(device)
        suffix_ids = full_ids[:, suffix_slice.start:suffix_slice.stop].to(device=device, dtype=torch.long).squeeze(0)  # shape = [suffix_len]
        gradients = self.llmlingua1_gradients(self.model, full_ids, suffix_slice, target_slice, device)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0) # avoid nan and inf shape = [suffix_len, vocab_size]

        # sample the sample_batch_size candidates from the top-k*suffix_len candidates
        candidates = self._sample_control(
            control_toks=suffix_ids,
            grad=gradients,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        # ------- evaluate the candidates -------
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)

        C = candidates.size(0)
        eval_full_ids = full_ids.expand(C, -1).clone()  # [C, sequence_length]
        eval_full_ids[:, suffix_slice] = candidates
        eval_output = self.model(input_ids=eval_full_ids)
        eval_logits = eval_output.logits # shape = [C, sequence_length, vocab_size]
        min_loss, min_idx = self.evaluate_candidates(eval_logits, eval_full_ids, suffix_slice, target_slice)
        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """Uniform interface for the COMA orchestrator. Unwraps single prompt."""
        return self.attack(prompts[0], guardrail_sentences[0], guardrail_keywords[0])


class AttackforLLMLingua2(object):
    """
    A class to attack the llmlingua2
    """
    def __init__(self, config: Optional[AttackConfig] = None, *, model=None,
                 tokenizer=None, rank_tokenizer=None,
                 compression_rates=(0.5, 0.6, 0.7), **config_kwargs):
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = AttackConfig(**config_kwargs)
        else:
            raise ValueError("AttackforLLMLingua2 requires config=AttackConfig(model_name='...') or model_name='...'")
        self.model_name = self.config.model_name
        load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model if model is not None else
                      AutoModelForTokenClassification.from_pretrained(self.model_name)).to(load_device)
        self.tokenizer = configure_right_padding(
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, return_offset_mapping=True,
            )
        )
        self.nonascii_toks = get_nonascii_toks_stable(self.tokenizer)

        self.device = self.model.device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.config.seed)
        self.vocab_size = get_embedding_matrix(self.model).shape[0]
        self.best_loss = float('inf')
        self.best_candidates = None
        self.last_step_loss = float('inf')
        self.last_step_metrics = None
        self.best_metrics = None
        self.loss_weight1 = getattr(self.config, 'loss_weight1', 1.0)
        self.loss_weight2 = getattr(self.config, 'loss_weight2', 1.0)
        self.rank_tokenizer = rank_tokenizer
        self.compression_rates = tuple(compression_rates)
        extras = getattr(self.config, "_extra", {})
        self.target_loss_weight = float(getattr(
            self.config, "target_loss_weight", extras.get("target_loss_weight", 0.0)))
        self.margin_hinge_weight = float(getattr(
            self.config, "margin_hinge_weight", extras.get("margin_hinge_weight", 0.0)))
        self.margin_delta = float(getattr(
            self.config, "margin_delta", extras.get("margin_delta", 0.0)))
        self.coordinate_width = int(getattr(
            self.config, "coordinate_width", extras.get("coordinate_width", 1)))
        self.require_clean_target_retention = bool(getattr(
            self.config, "require_clean_target_retention",
            extras.get("require_clean_target_retention", False)))
        if self.target_loss_weight < 0 or self.margin_hinge_weight < 0 or self.margin_delta < 0:
            raise ValueError("composite loss weights and margin_delta must be non-negative")
        if self.coordinate_width not in (1, 2):
            raise ValueError("coordinate_width must be 1 or 2")
        if not self.compression_rates or any(not 0 < rate < 1 for rate in self.compression_rates):
            raise ValueError("compression_rates must contain values between zero and one")

        # Surrogate CausalLM for PPL-based candidate scoring
        surrogate_name = getattr(self.config, 'surrogate_model_name', None)
        if surrogate_name:
            surrogate_dtype = torch.float16 if load_device.type == "cuda" else torch.float32
            self.surrogate_model = AutoModelForCausalLM.from_pretrained(
                surrogate_name, torch_dtype=surrogate_dtype,
            ).to(load_device)
            self.surrogate_tokenizer = AutoTokenizer.from_pretrained(
                surrogate_name, use_fast=True,
            )
            configure_right_padding(self.surrogate_tokenizer)
            if self.surrogate_tokenizer.pad_token is None:
                self.surrogate_tokenizer.pad_token = self.surrogate_tokenizer.eos_token
        else:
            self.surrogate_model = None
            self.surrogate_tokenizer = None

    def _starts_word(self, token: str, sentencepiece: bool) -> bool:
        if token.startswith("##"):
            return False
        if sentencepiece:
            return token.startswith(("▁", "Ġ")) or token in string.punctuation
        return True

    def _rank_token_count(self, tokens: List[str]) -> int:
        if self.rank_tokenizer is None:
            return 1
        # LLMLingua-2 ranks the merged XLM-R word with its leading ``▁``
        # marker intact; converting to a sentence first silently removes it.
        word = "".join(
            token[2:] if token.startswith("##") else token for token in tokens
        )
        try:
            return max(1, len(self.rank_tokenizer.encode(word, add_special_tokens=False)))
        except TypeError:
            return max(1, len(self.rank_tokenizer.encode(word)))

    def _budget_margin_metrics(self, logits, ids, attention_mask, target_mask):
        """Return the objective and the raw LLMLingua-2 budget diagnostics."""
        keep_probs = torch.softmax(logits, dim=-1)[..., 1]
        special_ids = set(self.tokenizer.all_special_ids)
        target_loss_weight = getattr(self, "target_loss_weight", 0.0)
        margin_hinge_weight = getattr(self, "margin_hinge_weight", 0.0)
        margin_delta = getattr(self, "margin_delta", 0.0)
        objectives, target_keep_scores, signed_margins, worst_margins = [], [], [], []
        for row in range(ids.size(0)):
            row_ids = ids[row].detach().cpu().tolist()
            row_attention = attention_mask[row].detach().cpu().bool().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(row_ids)
            active_tokens = [token for token, token_id, active in zip(tokens, row_ids, row_attention)
                             if active and token_id not in special_ids]
            sentencepiece = any(token.startswith(("▁", "Ġ")) for token in active_tokens)
            groups = []
            for position, (token, token_id, active) in enumerate(zip(tokens, row_ids, row_attention)):
                if not active or token_id in special_ids:
                    continue
                if not groups or self._starts_word(token, sentencepiece):
                    groups.append(([position], [token]))
                else:
                    groups[-1][0].append(position)
                    groups[-1][1].append(token)
            if not groups:
                raise ValueError("LLMLingua-2 budget loss found no content words")
            word_scores = torch.stack([
                keep_probs[row, positions].mean() for positions, _ in groups
            ])
            weights = torch.tensor([
                self._rank_token_count(group_tokens) for _, group_tokens in groups
            ], device=word_scores.device)
            ranked_scores = torch.repeat_interleave(word_scores, weights)
            target_positions = target_mask[row].bool()
            target_scores = [word_scores[index] for index, (positions, _) in enumerate(groups)
                             if target_positions[positions].any()]
            if not target_scores:
                raise ValueError("LLMLingua-2 budget loss found no target word")
            # The compressed phrase is absent as soon as any constituent word
            # is removed, so the weakest target word controls the margin.
            target_score = torch.stack(target_scores).min()
            margins = torch.stack([
                target_score - torch.quantile(ranked_scores, 1.0 - rate + 0.01)
                for rate in self.compression_rates
            ])
            worst_margin = margins.max()
            # Optional composite objective: lower the target's absolute keep
            # probability while requiring a configurable negative margin.
            # Defaults remain the paper-aligned raw margin objective.
            objective = worst_margin
            if target_loss_weight:
                objective = objective + target_loss_weight * target_score
            if margin_hinge_weight:
                objective = objective + margin_hinge_weight * torch.relu(
                    worst_margin + margin_delta)
            objectives.append(objective)
            target_keep_scores.append(target_score)
            signed_margins.append(margins)
            worst_margins.append(worst_margin)
        return {
            "objective_loss": torch.stack(objectives),
            "target_keep_score": torch.stack(target_keep_scores),
            "signed_margins": torch.stack(signed_margins),
            "worst_signed_margin": torch.stack(worst_margins),
        }

    def _budget_margin_losses(self, logits, ids, attention_mask, target_mask):
        """Compatibility wrapper returning only the optimized objective."""
        return self._budget_margin_metrics(
            logits, ids, attention_mask, target_mask)["objective_loss"]

    def llmlingua2_loss(self, logits, suffix_slice, target_slice, ids):
        """Official COMA optimization loss: classify target tokens as removed."""
        del suffix_slice
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        target_logits = logits[:, target_slice, :]
        labels = torch.zeros(target_logits.shape[:-1], dtype=torch.long,
                             device=target_logits.device)
        cls_loss = self.loss_weight1 * F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)), labels.reshape(-1)
        )
        if self.margin_hinge_weight or self.target_loss_weight:
            ids2 = ids if ids.dim() == 2 else ids.unsqueeze(0)
            attention_mask = torch.ones_like(ids2)
            target_mask = torch.zeros_like(ids2, dtype=torch.bool)
            target_mask[:, target_slice] = True
            raw = self._budget_margin_metrics(logits, ids2, attention_mask, target_mask)
            cls_loss = cls_loss + self.target_loss_weight * raw["target_keep_score"].mean()
            cls_loss = cls_loss + self.margin_hinge_weight * torch.relu(
                raw["worst_signed_margin"].mean() + self.margin_delta)
        return cls_loss

    def llmlingua2_gradients(
        self,
        model: AutoModelForTokenClassification,
        full_ids: torch.Tensor,
        suffix_slice: slice,
        target_slice: slice,
        device: torch.device,
    ):
        """
        Compute gradients of the LLMLingua2 loss w.r.t. suffix embeddings,
        then project back to vocab space via W^T.
        Returns gradient tensor of shape [suffix_len, vocab_size].
        """
        full_ids = full_ids.squeeze(0) # shape = [T]
        W = get_embedding_matrix(model) # shape = [vocab, dim]

        base_embeds = get_token_embeddings(model, full_ids.unsqueeze(0)).detach() # shape = [1, T, dim]

        # Create a differentiable copy of the suffix embeddings and substitute into full_embeds
        suffix_embeds = base_embeds[:, suffix_slice.start:suffix_slice.stop, :].clone().detach().requires_grad_(True)

        full_embeds = torch.cat([
            base_embeds[:, :suffix_slice.start, :],
            suffix_embeds,
            base_embeds[:, suffix_slice.stop:, :],
        ], dim=1)

        model.zero_grad(set_to_none=True)
        output = model(inputs_embeds=full_embeds, input_ids=None)
        logits = output.logits # shape = [1, T, num_labels]
        loss = self.llmlingua2_loss(
            logits=logits,
            suffix_slice=suffix_slice,
            target_slice=target_slice,
            ids=full_ids,
        )
        loss.backward()
        grad_one_hot = suffix_embeds.grad.squeeze(0) @ W.float().t() # shape = [suffix_len, vocab_size]
        return grad_one_hot


    def _sample_control(self, control_toks, grad, sample_batch_size, topk=256, sample_temp=1.0, generator=None):
        """Delegate to standalone sample_control."""
        candidates = sample_control(
            control_toks=control_toks, grad=grad,
            nonascii_toks=self.nonascii_toks, vocab_size=self.vocab_size,
            sample_batch_size=sample_batch_size, topk=topk,
            sample_temp=sample_temp, generator=generator,
        )
        if getattr(self, "coordinate_width", 1) == 2:
            candidates = self._add_second_coordinate(candidates, control_toks)
        return candidates

    @staticmethod
    def _add_second_coordinate(candidates, incumbent):
        """Turn half of one-token proposals into two-token proposals in-place."""
        if incumbent.numel() < 2 or candidates.size(0) < 2:
            return candidates
        changed = candidates.ne(incumbent).float().argmax(dim=1)
        has_change = candidates.ne(incumbent).any(dim=1)
        widened = candidates.clone()
        for row in range(0, candidates.size(0), 2):
            if not has_change[row]:
                continue
            for offset in range(1, candidates.size(0)):
                donor = (row + offset) % candidates.size(0)
                donor_pos = changed[donor]
                if has_change[donor] and donor_pos != changed[row]:
                    widened[row, donor_pos] = candidates[donor, donor_pos]
                    break
        return widened

    def _roundtrip_stable_candidates(self, candidates, fallback):
        stable = [
            self.tokenizer.encode(
                self.tokenizer.decode(row, skip_special_tokens=True).strip(),
                add_special_tokens=False,
            ) == row
            for row in candidates.detach().cpu().tolist()
        ]
        if any(stable):
            return candidates[torch.tensor(stable, device=candidates.device)]
        return fallback.unsqueeze(0)

    def _per_candidate_budget_metrics(self, eval_logits, eval_full_ids, suffix_slice,
                                      target_slice):
        """Compute objective and raw budget diagnostics for every candidate."""
        del suffix_slice
        eval_batch_size = self.config.eval_batch_size
        batch_num = math.ceil(eval_full_ids.size(0) / eval_batch_size)
        metrics = {}
        with torch.no_grad():
            for i in range(batch_num):
                batch_full_ids = eval_full_ids[i*eval_batch_size: (i+1)*eval_batch_size, :]
                batch_eval_logits = eval_logits[i*eval_batch_size: (i+1)*eval_batch_size, :, :]
                attention_mask = torch.ones_like(batch_full_ids)
                target_mask = torch.zeros_like(batch_full_ids, dtype=torch.bool)
                target_mask[:, target_slice] = True
                batch_metrics = self._budget_margin_metrics(
                    batch_eval_logits, batch_full_ids, attention_mask, target_mask)
                target_logits = batch_eval_logits[:, target_slice, :]
                labels = torch.zeros(target_logits.shape[:-1], dtype=torch.long,
                                     device=target_logits.device)
                cls_loss = self.loss_weight1 * F.cross_entropy(
                    target_logits.reshape(-1, target_logits.size(-1)),
                    labels.reshape(-1), reduction="none",
                ).reshape(target_logits.size(0), -1).mean(dim=1)
                batch_metrics["objective_loss"] = cls_loss
                if self.target_loss_weight:
                    batch_metrics["objective_loss"] = batch_metrics["objective_loss"] + (
                        self.target_loss_weight * batch_metrics["target_keep_score"])
                if self.margin_hinge_weight:
                    batch_metrics["objective_loss"] = batch_metrics["objective_loss"] + (
                        self.margin_hinge_weight * torch.relu(
                            batch_metrics["worst_signed_margin"] + self.margin_delta))
                for name, values in batch_metrics.items():
                    metrics.setdefault(name, []).append(values.detach().cpu())
        return {name: torch.cat(values, dim=0).numpy()
                for name, values in metrics.items()}

    def _per_candidate_cls_losses(self, eval_logits, eval_full_ids, suffix_slice, target_slice):
        """Compatibility wrapper returning the per-candidate objective."""
        return self._per_candidate_budget_metrics(
            eval_logits, eval_full_ids, suffix_slice, target_slice)["objective_loss"]

    def evaluate_candidates(self, eval_logits, eval_full_ids, suffix_slice, target_slice):
        """
        Pick best candidate based on classification loss only.
        """
        losses = self._per_candidate_cls_losses(eval_logits, eval_full_ids, suffix_slice, target_slice)
        min_idx = int(np.argmin(losses))
        min_loss = float(losses[min_idx])
        return min_loss, min_idx

    def _surrogate_ppl_losses(self, candidate_texts: List[str]) -> np.ndarray:
        """
        Compute per-sample PPL using the surrogate CausalLM.
        Higher value = higher perplexity = suffix is harder to predict.
        Returns numpy array of shape [C].
        """
        device = self.surrogate_model.device
        enc = self.surrogate_tokenizer(
            candidate_texts, return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        eval_batch_size = self.config.eval_batch_size
        all_losses = []
        with torch.no_grad():
            for start in range(0, input_ids.size(0), eval_batch_size):
                batch_ids = input_ids[start:start + eval_batch_size]
                batch_mask = attention_mask[start:start + eval_batch_size]
                output = self.surrogate_model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = output.logits  # [B, T, V]
                shift_logits = logits[:, :-1, :]
                shift_labels = batch_ids[:, 1:]
                shift_mask = batch_mask[:, 1:].float()
                per_tok = F.cross_entropy(
                    shift_logits.transpose(1, 2), shift_labels, reduction='none',
                )  # [B, T-1]
                per_sample = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)
                all_losses.append(per_sample.cpu())
        return torch.cat(all_losses, dim=0).numpy()  # [C]

    def attack(self, prompt: str, guardrail_sentence: str, guardrail_keyword: str):
        """
        Run one COMA step attacking LLMLingua2 on a single prompt.
        """
        device = self.model.device
        suffix_slice, target_slice = find_slices_from_token(self.tokenizer, prompt, guardrail_sentence, guardrail_keyword, self.config.suffix_length)
        full_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # shape = [1, T]
        if self.best_candidates is not None:
            full_ids[:, suffix_slice] = self.best_candidates.to(device)
        suffix_ids = full_ids[:, suffix_slice.start:suffix_slice.stop].to(device=device, dtype=torch.long).squeeze(0)  # shape = [suffix_len]
        gradients = self.llmlingua2_gradients(self.model, full_ids, suffix_slice, target_slice, device)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0) # shape = [suffix_len, vocab_size]

        # sample the sample_batch_size candidates from the top-k*suffix_len candidates
        candidates = self._sample_control(
            control_toks=suffix_ids,
            grad=gradients,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )
        # Always score the incumbent alongside one-flip mutations.  Without
        # this, resetting best_loss for a new run discards a warm start on the
        # first step and can only move uphill before the search recovers.
        candidates = torch.cat([suffix_ids.unsqueeze(0), candidates], dim=0)

        # The deployed suffix crosses an IDs -> text -> IDs boundary.  Reject
        # candidates that change at that boundary before they can become the
        # optimizer's persistent best candidate.
        candidates = self._roundtrip_stable_candidates(candidates, suffix_ids)

        # ------- evaluate the candidates -------
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)

        C = candidates.size(0)
        eval_full_ids = full_ids.expand(C, -1).clone()  # [C, sequence_length]
        eval_full_ids[:, suffix_slice] = candidates
        eval_output = self.model(input_ids=eval_full_ids)
        eval_logits = eval_output.logits  # [C, T, num_labels]

        # Per-candidate cls loss
        candidate_metrics = self._per_candidate_budget_metrics(
            eval_logits, eval_full_ids, suffix_slice, target_slice)
        cls_losses = candidate_metrics["objective_loss"]  # [C]

        # Combine with surrogate PPL if available
        if self.surrogate_model is not None:
            candidate_texts = self.tokenizer.batch_decode(eval_full_ids, skip_special_tokens=True)
            ppl_losses = self._surrogate_ppl_losses(candidate_texts)  # [C]
            # Minimize cls_loss + w2*ppl_loss -> lower cls, lower PPL (target gets removed)
            total_losses = cls_losses + self.loss_weight2 * ppl_losses
        else:
            total_losses = cls_losses

        min_idx = int(np.argmin(total_losses))
        min_loss = float(total_losses[min_idx])
        self.last_step_loss = min_loss
        self.last_step_metrics = {
            "objective_loss": float(cls_losses[min_idx]),
            "target_keep_score": float(candidate_metrics["target_keep_score"][min_idx]),
            "signed_margins": candidate_metrics["signed_margins"][min_idx].tolist(),
            "compression_rates": list(self.compression_rates),
            "worst_signed_margin": float(
                candidate_metrics["worst_signed_margin"][min_idx]),
        }

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
            self.best_metrics = dict(self.last_step_metrics)
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """Uniform interface for the COMA orchestrator. Unwraps single prompt."""
        return self.attack(prompts[0], guardrail_sentences[0], guardrail_keywords[0])


def slice_to_pos(slices: List[slice], device: torch.device):
    """
    This function is used to convert the slices to the positions
    """
    B = len(slices)
    lens = [sl.stop - sl.start for sl in slices]
    if len(set(lens)) != 1:
        raise ValueError(f"All suffix lengths must be equal to share candidates. Got lengths={lens}")
    L = lens[0]

    pos = torch.empty(B, L, device=device, dtype=torch.long)
    for b, sl in enumerate(slices):
        pos[b] = torch.arange(sl.start, sl.stop, device=device, dtype=torch.long)
    return pos  # shape = [B, L]

class MultiplePromptsAttackforLLMlingua1(AttackforLLMLingua1):
    """
    A class to attack multiple prompts (shared suffix optimization).
    """
    def __init__(self, config: Optional[AttackConfig] = None, **config_kwargs):
        super().__init__(config=config, **config_kwargs)

    def batch_llmlingua1_loss(self, shift_logits: torch.Tensor, shift_labels: torch.Tensor, shift_mask: torch.Tensor):
        """
        This function is used to batch the llmlingua1 loss
        """
        per_tok = -1.0 * F.cross_entropy(
            shift_logits.transpose(1, 2),  # [B,vocab_size,T-1]
            shift_labels,                  # [B,T-1]
            reduction="none"
        )  # [B,T-1]

        per_ex = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0) # shape = [B]
        return per_ex

    def batch_llmlingua1_gradients(self, model, full_ids, suffix_slices, target_slices, attention_mask, device):
        """
        This function is used to batch the llmlingua1 gradients
        """
        model.zero_grad(set_to_none=True)
        full_ids = full_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)
        suffix_length = suffix_slices[0].stop - suffix_slices[0].start
        suffix_starts = torch.tensor([sl.start for sl in suffix_slices], device=device, dtype=torch.long)
        pos = suffix_starts[:, None] + torch.arange(suffix_length, device=device) # shape = [B, suffix_length]
        base_embeds = get_token_embeddings(model, full_ids).detach() # shape = [B, sequence_length, dim]
        B = full_ids.size(0)
        T = full_ids.size(1)
        W = get_embedding_matrix(model)
        V = W.size(0)
        D = base_embeds.size(-1)
        # start0, end0 = suffix_slices[0].start, suffix_slices[0].stop
        suffix_ids0 = full_ids[0,suffix_slices[0].start:suffix_slices[0].stop] # shape = [suffix_length]
        one_hot = torch.zeros((suffix_length, V), device=device, dtype=torch.float32)
        one_hot.scatter_(1, suffix_ids0.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        suffix_embeds = (one_hot @ W.float()).to(dtype=base_embeds.dtype) # shape = [suffix_length, dim]
        suffix_values = suffix_embeds.unsqueeze(0).expand(B, -1, -1)  # shape = [B, suffix_length, dim]
        index = pos.unsqueeze(-1).expand(B, suffix_length, D)  # shape = [B, suffix_length, D]
        suffix_full = torch.zeros(B, T, D, device=device, dtype=base_embeds.dtype)
        suffix_full.scatter_(1, index, suffix_values)  # shape = [B, T, dim]

        # get the suffix_mask: shape = [B, T]
        suffix_mask = torch.zeros_like(full_ids, dtype=torch.bool) # shape = [B, T]
        suffix_mask.scatter_(1, pos, torch.ones((B, suffix_length),device=device, dtype=torch.bool))
        m = suffix_mask.unsqueeze(-1).to(dtype=base_embeds.dtype) # shape = [B, T, 1]

        # replace the suffix with the suffix_full in the base_embeds
        full_embeds = base_embeds * (1.0 - m) + suffix_full # shape = [B, T, dim]

        # get the outputs
        logits = model(inputs_embeds=full_embeds, attention_mask=attention_mask).logits # shape = [B, T, vocab_size]

        # target_mask, in the attack for llmlingua1, the target_mask is the suffix_mask
        target_mask = suffix_mask & attention_mask.bool()     # [B,T]
        shift_logits = logits[:, :-1, :]            # [B,T-1,vocab_size]
        shift_labels = full_ids[:, 1:]              # [B,T-1] shape = [B, T-1]
        shift_mask  = target_mask[:, 1:].float()    # [B,T-1]

        loss = self.batch_llmlingua1_loss(shift_logits, shift_labels, shift_mask)
        loss = loss.mean() # shape = []
        loss.backward()
        return one_hot.grad, target_mask # shape = [suffix_length, vocab_size]

    def batch_eval_candidates(self, model, eval_full_ids: torch.Tensor, eval_attention_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        Evaluate candidates: eval_full_ids [C*B, T], aggregate per-candidate loss over B prompts.
        target_mask [B, T] is expanded to [C*B, T] to align with eval_full_ids.
        Returns min_loss, min_idx over C candidates.
        """
        with torch.no_grad():
            C = self.config.sample_batch_size
            B = target_mask.size(0)
            # Expand target_mask to match eval layout [C*B, T]
            eval_target_mask = target_mask.unsqueeze(0).expand(C, -1, -1).reshape(C * B, -1)
            eval_batch_size = self.config.eval_batch_size
            all_losses = []
            for start in range(0, eval_full_ids.size(0), eval_batch_size):
                batch_ids = eval_full_ids[start:start + eval_batch_size, :]
                batch_mask = eval_attention_mask[start:start + eval_batch_size, :]
                output = model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = output.logits  # [batch, T, vocab_size]
                shift_logits = logits[:, :-1, :]
                shift_labels = batch_ids[:, 1:]
                shift_mask = eval_target_mask[start:start + eval_batch_size, 1:].float()
                loss = self.batch_llmlingua1_loss(shift_logits, shift_labels, shift_mask)
                all_losses.append(loss)
            per_sample_losses = torch.cat(all_losses, dim=0)  # [C*B]
            # Reshape: [C, B], mean over B -> [C]
            per_sample_losses = per_sample_losses.reshape(C, B)
            per_candidate_loss = per_sample_losses.mean(dim=1)
            min_idx = int(per_candidate_loss.argmin().item())
            min_loss = float(per_candidate_loss[min_idx])
            return min_loss, min_idx

    def build_eval_full_ids(self, full_ids, candidates, suffix_slices, attention_mask, device):
        """
        This function is used to build the eval_full_ids
        """
        full_ids = full_ids.to(device=device, dtype=torch.long) # shape = [B,T], B = len of prompt_batch
        candidates = candidates.to(device=device, dtype=torch.long) # shape = [C, suffix_length], C = sample_batch_size
        B, T = full_ids.shape
        C, L= candidates.shape

        eval_full_ids_origin = full_ids.unsqueeze(0).expand(C, -1, -1).clone() # shape = [C, B, T]
        eval_full_ids = eval_full_ids_origin.reshape((C*B, T)) # shape = [C*B, T]

        pos = slice_to_pos(suffix_slices, device) # shape = [B, L]

        pos_repeat = pos.unsqueeze(0).expand(C,-1,-1).reshape((C*B, L)) # shape = [C*B, L]
        candidate_repeat = candidates.unsqueeze(1).expand(C,B,L).reshape((C*B, L)) # shape = [C*B, L]
        rows = torch.arange(C*B, device=device, dtype=torch.long).unsqueeze(1).expand(C*B, L) # shape = [C*B, L]
        eval_full_ids[rows, pos_repeat] = candidate_repeat # shape = [C*B, T]
        eval_attention_mask = attention_mask.unsqueeze(0).expand(C, -1, -1).reshape((C*B, T))
        return eval_full_ids, eval_full_ids_origin, eval_attention_mask # shape = [C*B, T], shape = [C, B, T], shape = [C*B, T]

    def attack(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """
        Run one COMA step attacking LLMLingua1 across multiple prompts with shared suffix.
        """
        validate_prompt_batch(prompts, guardrail_sentences, guardrail_keywords)
        suffix_slices, target_slices = [], []
        for prompt, guardrail_sentence, guardrail_keyword in zip(prompts, guardrail_sentences, guardrail_keywords):
            suffix_slice, target_slice = find_slices_from_token(self.tokenizer, prompt, guardrail_sentence, guardrail_keyword, self.config.suffix_length)
            suffix_slices.append(suffix_slice)
            target_slices.append(target_slice)
        device = self.model.device
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        full_ids = enc.input_ids.to(device=device)  # shape = [B, T]: B = N
        attention_mask = enc.attention_mask.to(device=device)
        suffix_ids = full_ids[0, suffix_slices[0].start:suffix_slices[0].stop].clone()  # shape = [suffix_len]
        gradients, target_mask = self.batch_llmlingua1_gradients(self.model, full_ids, suffix_slices, target_slices, attention_mask, device)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)  # shape = [suffix_len, vocab_size]

        candidates = self._sample_control(
            control_toks=suffix_ids,
            grad=gradients,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        # ------- evaluate the candidates -------
        eval_full_ids, eval_full_ids_origin, eval_attention_mask = self.build_eval_full_ids(full_ids, candidates, suffix_slices, attention_mask, device)
        min_loss, min_idx = self.batch_eval_candidates(self.model, eval_full_ids, eval_attention_mask, target_mask)
        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """Uniform interface for the COMA orchestrator. Passes lists directly."""
        return self.attack(prompts, guardrail_sentences, guardrail_keywords)

class MultiplePromptsAttackforLLMlingua2(AttackforLLMLingua2):
    """
    A class to attack multiple prompts (shared suffix optimization).
    """
    def __init__(self, config: Optional[AttackConfig] = None, **config_kwargs):
        super().__init__(config=config, **config_kwargs)

    def batch_llmlingua2_loss(self, shift_logits: torch.Tensor, shift_labels: torch.Tensor, shift_mask: torch.Tensor):
        """
        Batch LLMLingua2 loss: token classification (predict remove=0 at suffix) + optional PPL.
        shift_logits: [B, T-1, num_labels] for token cls, or [B, T-1, vocab] if joint model
        """
        # Token cls: predict class 0 (remove) at suffix positions
        token_cls_labels = torch.full(shift_logits.shape[:2], -100, device=shift_logits.device, dtype=torch.long)
        token_cls_labels[shift_mask.bool()] = 0
        loss1 = F.cross_entropy(
            shift_logits.transpose(1, 2), token_cls_labels, reduction="none", ignore_index=-100
        )  # [B, T-1]
        loss1 = torch.nan_to_num(loss1, nan=0.0, posinf=0.0, neginf=0.0)

        # PPL: only if logits have vocab size (joint model)
        if shift_logits.size(-1) > 2:
            loss2 = F.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none"
            )
            per_tok = self.loss_weight1 * loss1 + self.loss_weight2 * loss2
        else:
            per_tok = self.loss_weight1 * loss1

        per_ex = (per_tok * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp_min(1.0)  # shape = [B]
        return per_ex

    def batch_llmlingua2_gradients(
        self,
        model,
        full_ids,
        suffix_slices,
        target_slices,
        attention_mask,
        device
    ):
        """
        Compute gradients w.r.t. suffix embeddings directly (continuous relaxation),
        then project to vocab space via W^T and aggregate over batch.
        Returns:
            grad: [suffix_length, vocab_size]
            target_mask: [B, T] (bool)
        """
        model.zero_grad(set_to_none=True)

        full_ids = full_ids.to(device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device=device)  # [B,T], 1=valid, 0=pad

        B, T = full_ids.shape

        # All prompts in batch must share the same suffix_length
        suffix_length = suffix_slices[0].stop - suffix_slices[0].start

        suffix_starts = torch.tensor(
            [sl.start for sl in suffix_slices],
            device=device,
            dtype=torch.long
        )  # [B]

        # pos: [B, L], each row is the suffix token position indices for that sample
        pos = suffix_starts[:, None] + torch.arange(suffix_length, device=device)[None, :]  # [B,L]

        # [B,T,D], detach to avoid backprop through the embedding table
        base_embeds = get_token_embeddings(model, full_ids).detach()
        D = base_embeds.size(-1)

        # 1) Gather suffix embeddings (supports different start per sample)
        gather_index = pos.unsqueeze(-1).expand(B, suffix_length, D)  # [B,L,D]
        suffix_init = base_embeds.gather(dim=1, index=gather_index)   # [B,L,D]

        # 2) Make suffix_embeds a leaf tensor for gradient computation
        suffix_embeds = suffix_init.detach().clone().requires_grad_(True)  # [B,L,D]

        # 3) Scatter suffix_embeds back into base_embeds (out-of-place for autograd)
        full_embeds = base_embeds.scatter(dim=1, index=gather_index, src=suffix_embeds)  # [B,T,D]

        # 4) Forward pass (pass attention_mask to avoid padding contamination)
        logits = model(inputs_embeds=full_embeds, attention_mask=attention_mask).logits  # [B,T,V]

        # Token-classification labels are aligned with their input token; the
        # deletion objective therefore belongs on target_slices, not the query
        # suffix and not a next-token-shifted position.
        target_mask = torch.tensor(
            target_span_mask(target_slices, attention_mask.detach().cpu().tolist()),
            device=device,
            dtype=torch.bool,
        )
        loss = self.batch_llmlingua2_loss(logits, full_ids, target_mask.float())
        if loss.ndim > 0:  # per-example vector -> mean
            loss = loss.mean()

        loss.backward()
        W = get_embedding_matrix(model).detach()
        grad = suffix_embeds.grad @ W.float().t()  # shape = [B, suffix_length, vocab_size]
        # Aggregate over batch for shared suffix sampling
        grad = grad.mean(dim=0)  # shape = [suffix_length, vocab_size]
        return grad, target_mask

    def _batch_per_candidate_cls_losses(self, model, eval_full_ids: torch.Tensor, eval_attention_mask: torch.Tensor, target_mask: torch.Tensor) -> np.ndarray:
        """
        Compute per-candidate cls loss (averaged over B prompts). Returns numpy array [C].
        """
        with torch.no_grad():
            C = self.config.sample_batch_size
            B = target_mask.size(0)
            eval_target_mask = target_mask.unsqueeze(0).expand(C, -1, -1).reshape(C * B, -1)
            eval_batch_size = self.config.eval_batch_size
            all_losses = []
            for start in range(0, eval_full_ids.size(0), eval_batch_size):
                batch_ids = eval_full_ids[start:start + eval_batch_size, :]
                batch_mask = eval_attention_mask[start:start + eval_batch_size, :]
                output = model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = output.logits
                batch_target_mask = eval_target_mask[start:start + eval_batch_size].float()
                loss = self.batch_llmlingua2_loss(logits, batch_ids, batch_target_mask)
                all_losses.append(loss)
            per_sample_losses = torch.cat(all_losses, dim=0)  # [C*B]
            per_sample_losses = per_sample_losses.reshape(C, B)
            per_candidate_loss = per_sample_losses.mean(dim=1)  # [C]
            return per_candidate_loss.cpu().numpy()

    def batch_eval_candidates(self, model, eval_full_ids: torch.Tensor, eval_attention_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        Evaluate candidates: pick best based on cls loss only.
        """
        per_candidate_loss = self._batch_per_candidate_cls_losses(model, eval_full_ids, eval_attention_mask, target_mask)
        min_idx = int(np.argmin(per_candidate_loss))
        min_loss = float(per_candidate_loss[min_idx])
        return min_loss, min_idx

    def build_eval_full_ids(self, full_ids, candidates, suffix_slices, attention_mask, device):
        """
        This function is used to build the eval_full_ids
        """
        full_ids = full_ids.to(device=device, dtype=torch.long) # shape = [B,T], B = len of prompt_batch
        candidates = candidates.to(device=device, dtype=torch.long) # shape = [C, suffix_length], C = sample_batch_size
        B, T = full_ids.shape
        C, L= candidates.shape

        eval_full_ids_origin = full_ids.unsqueeze(0).expand(C, -1, -1).clone() # shape = [C, B, T]
        eval_full_ids = eval_full_ids_origin.reshape((C*B, T)) # shape = [C*B, T]

        pos = slice_to_pos(suffix_slices, device) # shape = [B, L]

        pos_repeat = pos.unsqueeze(0).expand(C,-1,-1).reshape((C*B, L)) # shape = [C*B, L]
        candidate_repeat = candidates.unsqueeze(1).expand(C,B,L).reshape((C*B, L)) # shape = [C*B, L]
        rows = torch.arange(C*B, device=device, dtype=torch.long).unsqueeze(1).expand(C*B, L) # shape = [C*B, L]
        eval_full_ids[rows, pos_repeat] = candidate_repeat # shape = [C*B, T]
        eval_attention_mask = attention_mask.unsqueeze(0).expand(C, -1, -1).reshape((C*B, T))
        return eval_full_ids, eval_full_ids_origin, eval_attention_mask # shape = [C*B, T], shape = [C, B, T], shape = [C*B, T]


    def attack(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """
        Run one COMA step attacking LLMLingua2 across multiple prompts with shared suffix.
        """
        validate_prompt_batch(prompts, guardrail_sentences, guardrail_keywords)
        suffix_slices, target_slices = [], []
        for prompt, guardrail_sentence, guardrail_keyword in zip(prompts, guardrail_sentences, guardrail_keywords):
            suffix_slice, target_slice = find_slices_from_token(self.tokenizer, prompt, guardrail_sentence, guardrail_keyword, self.config.suffix_length)
            suffix_slices.append(suffix_slice)
            target_slices.append(target_slice)
        device = self.model.device
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        full_ids = enc.input_ids.to(device=device)  # shape = [B, T]
        attention_mask = enc.attention_mask.to(device=device)
        if self.best_candidates is not None:
            best = self.best_candidates.to(device)
            for row, suffix_slice in enumerate(suffix_slices):
                full_ids[row, suffix_slice] = best
        suffix_ids = full_ids[0, suffix_slices[0].start:suffix_slices[0].stop].clone()  # shape = [suffix_len]
        gradients, target_mask = self.batch_llmlingua2_gradients(self.model, full_ids, suffix_slices, target_slices, attention_mask, device)
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)  # shape = [suffix_len, vocab_size]

        candidates = self._sample_control(
            control_toks=suffix_ids,
            grad=gradients,
            sample_batch_size=self.config.sample_batch_size,
            topk=self.config.top_k,
            generator=self.gen,
        )

        # ------- evaluate the candidates -------
        eval_full_ids, eval_full_ids_origin, eval_attention_mask = self.build_eval_full_ids(full_ids, candidates, suffix_slices, attention_mask, device)

        # Per-candidate cls loss (averaged over B prompts)
        cls_losses = self._batch_per_candidate_cls_losses(self.model, eval_full_ids, eval_attention_mask, target_mask)  # [C]

        # Combine with surrogate PPL if available
        if self.surrogate_model is not None:
            C = self.config.sample_batch_size
            B = len(prompts)
            # Decode all C*B candidate prompts, compute PPL, reshape and average over B
            candidate_texts = self.tokenizer.batch_decode(eval_full_ids, skip_special_tokens=True)  # [C*B]
            ppl_all = self._surrogate_ppl_losses(candidate_texts)  # [C*B]
            ppl_per_candidate = ppl_all.reshape(C, B).mean(axis=1)  # [C]
            total_losses = cls_losses + self.loss_weight2 * ppl_per_candidate
        else:
            total_losses = cls_losses

        min_idx = int(np.argmin(total_losses))
        min_loss = float(total_losses[min_idx])

        if min_loss < self.best_loss:
            self.best_loss = min_loss
            self.best_candidates = candidates[min_idx].clone()
        if self.best_candidates is None:
            self.best_candidates = candidates[0].clone()

        return self.best_loss, self.best_candidates.detach().cpu().numpy().tolist()

    def step(self, prompts: List[str], guardrail_sentences: List[str], guardrail_keywords: List[str]):
        """Uniform interface for the COMA orchestrator. Passes lists directly."""
        return self.attack(prompts, guardrail_sentences, guardrail_keywords)
