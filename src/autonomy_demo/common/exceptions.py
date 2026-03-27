class AutonomyDemoError(Exception):
    """Base error for the scaffold."""


class ConfigurationError(AutonomyDemoError):
    """Raised when runtime configuration is invalid."""


class ScenarioValidationError(AutonomyDemoError):
    """Raised when a scenario file fails schema validation."""


class CarlaRuntimeError(AutonomyDemoError):
    """Raised when the live CARLA runtime cannot be configured or reached."""
