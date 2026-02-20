from __future__ import annotations

from enum import Enum


ALL_GROUP_KEY = "__ALL__"


class ModelClass(str, Enum):
    ANOVA = "ANOVA"
    ANCOVA = "ANCOVA"


MODEL_CLASS_ANOVA = ModelClass.ANOVA.value
MODEL_CLASS_ANCOVA = ModelClass.ANCOVA.value


class ValidityClass(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    VALID_WITH_ROBUST_SE = "valid_with_robust_se"


VALIDITY_VALID = ValidityClass.VALID.value
VALIDITY_INVALID = ValidityClass.INVALID.value
VALIDITY_VALID_WITH_ROBUST_SE = ValidityClass.VALID_WITH_ROBUST_SE.value


class StabilityBucket(str, Enum):
    STABLE = "STABLE"
    CONDITIONAL = "CONDITIONAL"
    NON_EFFECT = "NON_EFFECT"
    REDFLAG = "REDFLAG"


BUCKET_STABLE = StabilityBucket.STABLE.value
BUCKET_CONDITIONAL = StabilityBucket.CONDITIONAL.value
BUCKET_NON_EFFECT = StabilityBucket.NON_EFFECT.value
BUCKET_REDFLAG = StabilityBucket.REDFLAG.value


GROUP_MISSING_POLICY_AS_LEVEL = "AS_LEVEL"
GROUP_MISSING_POLICY_DROP_ROWS = "DROP_ROWS"
GROUP_MISSING_LEVEL_TOKEN = "__NA__"


OUTLIER_DECISION_KEEP = "keep"
OUTLIER_DECISION_REMOVE = "remove"


ROBUST_SE_MODES = {"HC0", "HC1", "HC2", "HC3"}


ANALYSIS_STATUS_CONFIGURED = "configured"
ANALYSIS_STATUS_REGISTRY_GENERATED = "registry_generated"
ANALYSIS_STATUS_MODELS_RAN = "models_ran"
ANALYSIS_STATUS_STABILITY_COMPLETE = "stability_complete"


MODEL_REGISTRY_STATUS_QUEUED = "queued"
MODEL_REGISTRY_STATUS_COMPLETED = "completed"


MODEL_RUN_STATUS_COMPLETED = "completed"
MODEL_RUN_STATUS_FAILED = "failed"
