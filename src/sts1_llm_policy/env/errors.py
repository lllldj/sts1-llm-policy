"""Combat termination shared by live and simulator environments."""


class CombatEndedError(RuntimeError):
    """Raised when step() leaves the combat decision state, retaining raw output."""

    def __init__(self, raw_state: dict) -> None:
        super().__init__("Combat ended or game left COMBAT state")
        self.raw_state = raw_state
