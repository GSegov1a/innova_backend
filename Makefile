.PHONY: dev

dev:
	$(UVICORN) app.main:app --reload
