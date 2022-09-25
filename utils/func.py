def normalize_zero_to_one(data, eps=1e-6):
        data_min = float(data.min())
        data_max = float(data.max())
        return (data - data_min) / (data_max - data_min + eps)
