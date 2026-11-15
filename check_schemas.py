from schemas import DatasetRowChurn

row = DatasetRowChurn(
    monthly_fee=49.99,
    usage_hours=12.5,
    support_requests=3,
    account_age_months=18,
    failed_payments=1,
    region="europe",
    device_type="mobile",
    payment_method="card",
    autopay_enabled=1,
    churn=0,
)

print(row.model_dump())
