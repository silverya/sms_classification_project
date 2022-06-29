class PreprocessingManager():
    def __init__(
        self,
        feature_column_name: str,
        label_column_name: str,
    ) -> None:
        self.feature_column_name = feature_column_name
        self.label_column_name = label_column_name