"""P5i clean execution of P5h's guarded shared-dense protocol.

P5i changes neither the P5e frozen system nor P5h's execution mechanism.  It
only consumes new identities after the P5h pre-model field-name exception.
"""
from cj0724.p5h_shared_dense_execution import main


if __name__ == "__main__":
    main()
