.PHONY: install playground

install:
	agents-cli install

playground:
	uv run uvicorn expense_agent.fast_api_app:app --host 127.0.0.1 --port 8080
