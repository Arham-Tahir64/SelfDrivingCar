class AutonomyDemoError(Exception):
    """Base error for the scaffold."""


class ConfigurationError(AutonomyDemoError):
    """Raised when runtime configuration is invalid."""


class ScenarioValidationError(AutonomyDemoError):
    """Raised when a scenario file fails schema validation."""

