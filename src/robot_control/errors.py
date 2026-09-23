class PiperControlError(RuntimeError):
    """Base error for the unified control layer."""


class BackendUnavailableError(PiperControlError):
    pass


class IKError(PiperControlError):
    pass


class NotConnectedError(PiperControlError):
    pass
