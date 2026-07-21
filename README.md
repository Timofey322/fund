# Fund — quantitative trading pipeline

```powershell
cd fund
python run.py pipeline run --checkpoint
python run.py verify
```

## Setup

1. Copy `.env.example` to `.env` and set `FINAM_API_SECRET` if using Finam.
2. `pip install -r requirements-quant.txt`
3. Run verify, then pipeline.

Outputs live under `../output/`; market data under `../data/`.
