"""Pipeline integrity verification suite and design checks."""

from hargap.verify.checks import (
    permutation_check,
    reproduction_check,
    identity_probe_check,
)
from hargap.verify.suite import run_verification_suite

# Spec-compatible alias
run_suite = run_verification_suite

__all__ = [
    "permutation_check",
    "reproduction_check",
    "identity_probe_check",
    "run_verification_suite",
    "run_suite",
]
