# DeepSeek Harness — Pure Python Rewrite

This branch (`experiment/python-rewrite`) is a from-scratch Python port of [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness). The TypeScript monorepo is not present on this branch; use `git show upstream/master:<path>` to consult the original sources.

Porting baseline: vendored Cordis 4.0.0-rc.7 (`cordiverse/cordis` @ `56b3d4f`) plus the local modifications logged in upstream `vendor/README.md`.

## Layout

- `src/pycordis/` — Python port of the Cordis framework kernel (context / reflect / registry / fiber / events / service / logger)
- `src/pydsh/` — harness capability seams (fs / subprocess / llm / llm_deepseek / session / system_prompt / agent)
- `tests/` — behavioral tests

## Development

```sh
uv venv .venv && uv pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
.venv/bin/mypy src
```

MIT-licensed, as is the upstream project. See `LICENSE`.
