import polars as pl

class Files:
    def __init__(self) -> None:
        self.path_file_csv = "files_csv/input_old_files.csv"
        self.improved_file_csv = "files_csv/output_file.csv"
        self.df_default = self.read_csv() # Return my df in instance of class
    
    def read_csv(self) -> pl.DataFrame:
        df = pl.read_csv(self.path_file_csv)
        return df
    

    def create_csv(self, dataFrame: pl.DataFrame) -> pl.DataFrame:
        return dataFrame.write_csv(self.improved_file_csv)
    
    
    def filter_items(self, bosch_items: bool) -> pl.DataFrame:
        filter_logic = pl.col("s_name").str.contains("Bosch Rexroth Ltda.", literal=True)
        return self.df_default.filter(filter_logic if bosch_items else ~filter_logic)
    
    def bosch_items(self) -> pl.DataFrame:
        return self.filter_items(bosch_items=True)
    
    def no_bosch_items(self) -> pl.DataFrame:
        return self.filter_items(bosch_items=False)