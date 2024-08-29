trade_schema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                    "ticker": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["date", "price", "ticker"]
            },
            "minItems": 10
        }
    },
    "required": ["data"]
}