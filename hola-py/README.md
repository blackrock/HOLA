# hola-opt

`hola-opt` is the Python interface to HOLA (Hyperparameter Optimization,
Lightweight Asynchronous), a black-box optimization engine backed by Rust.
It provides the same ask/tell workflow for local studies and remote HOLA
servers.

```python
from hola_opt import Minimize, Real, Space, Study

study = Study(
    space=Space(x=Real(0.0, 1.0)),
    objectives=[Minimize("loss")],
    strategy="sobol",
    seed=42,
)
trial = study.ask()
study.tell(trial.trial_id, {"loss": trial.params["x"] ** 2})
```

See the [HOLA documentation](https://github.com/blackrock/HOLA/blob/main/docs/index.md)
for installation, complete examples, distributed workers, and the REST API.

HOLA is licensed under the Apache License 2.0. The license text is included
with source and wheel distributions.
