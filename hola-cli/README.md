# hola-cli

`hola-cli` provides the `hola` command-line program for running a HOLA REST
server and distributed workers.

```bash
hola serve example_study.yaml --port 8000
hola worker --server http://localhost:8000 --exec "python train.py"
```

See the [CLI and distributed-operation guide](https://github.com/blackrock/HOLA/blob/main/docs/cli-guide.md)
for configuration, authentication, checkpointing, and worker modes.

Licensed under the Apache License 2.0. The complete license text is included
in `LICENSE-APACHE`.
