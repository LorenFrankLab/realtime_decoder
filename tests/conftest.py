"""Shared test setup.

Several realtime_decoder modules (base, position, synthetic, ...) import
`mpi4py` at module top. mpi4py is non-trivial to install in a CI image
(needs a system MPI build), and none of the deterministic logic we want
to unit-test actually exercises it. We stub it here so importing
realtime_decoder modules works on a plain `pip install pytest` env.

If real mpi4py is already importable (developer machine with MPI
installed), we skip the stub and use the real module.
"""

import sys
import types


def _install_mpi4py_stub():
    if 'mpi4py' in sys.modules:
        return
    try:
        import mpi4py  # noqa: F401
        return
    except ImportError:
        pass

    mpi4py_mod = types.ModuleType('mpi4py')
    mpi_mod = types.ModuleType('mpi4py.MPI')

    class _Status:  # minimal stand-in for MPI.Status()
        source = 0
        tag = 0

    class _Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            pass

        # the data-path methods are not exercised by any tests that
        # touch this stub; leave them as raising stubs to surface
        # accidental hot-path use.
        def Send(self, *a, **k):
            raise RuntimeError("MPI stub: Send called in a unit test")

        def Irecv(self, *a, **k):
            raise RuntimeError("MPI stub: Irecv called in a unit test")

        def send(self, *a, **k):
            raise RuntimeError("MPI stub: send called in a unit test")

        def irecv(self, *a, **k):
            raise RuntimeError("MPI stub: irecv called in a unit test")

    mpi_mod.Status = _Status
    mpi_mod.Comm = _Comm
    mpi_mod.BYTE = object()  # sentinel; tests don't inspect it
    mpi_mod.COMM_WORLD = _Comm()
    mpi4py_mod.MPI = mpi_mod
    sys.modules['mpi4py'] = mpi4py_mod
    sys.modules['mpi4py.MPI'] = mpi_mod


_install_mpi4py_stub()
