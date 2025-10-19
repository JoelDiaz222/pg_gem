```
python3 -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
pip install grpcio grpcio-tools sentence-transformers asyncio
python3 embedding_api_grpc.py
```

```
cd proto
python -m grpc_tools.protoc -I. --python_out=.. --grpc_python_out=.. tei.proto
```