#!/bin/bash
uvicorn --host 0.0.0.0 --port $PORT --reload backend.main:app
#uvicorn --host 0.0.0.0 --port 8090 --reload backend.main:app 
