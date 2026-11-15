.PHONY: test server ui dev

test:
	pytest

server:
	langgraph dev --host 127.0.0.1 --port 2024 --no-browser --allow-blocking

ui:
	cd ui && pnpm dev

dev:
	./scripts/dev.sh
