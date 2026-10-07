"""Shared exception identity across imports and python -m entrypoints."""


class Stop(RuntimeError):
    def __init__(self,status,reason):
        super().__init__(reason)
        self.status=status
