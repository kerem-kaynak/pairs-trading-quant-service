rlrt_schema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "format": "date"},
                    "spread": {"type": "number"}
                },
                "required": ["date", "spread"]
            },
            "minItems": 10
        }
    },
    "required": ["data"]
}

pairs_schema = {
    "type": "object",
    "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "date": {"type": "string", "format": "date"},
                    "price": {"type": "number"}
                },
                "required": ["ticker", "date", "price"],
                "additionalProperties": True
            },
            "minItems": 1
        }
    },
    "required": ["data"]
}