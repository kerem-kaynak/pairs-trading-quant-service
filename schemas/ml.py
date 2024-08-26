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