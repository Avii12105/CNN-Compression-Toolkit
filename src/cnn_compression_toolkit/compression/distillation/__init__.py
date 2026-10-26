"""Knowledge-distillation utilities."""

from .distiller import DistillationResult, distill_student, distillation_loss

__all__ = ["DistillationResult", "distillation_loss", "distill_student"]