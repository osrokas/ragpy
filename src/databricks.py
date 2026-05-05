
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import expr

class PDFLoader:
    def __init__(self, spark: SparkSession, path: str):
        self.spark = spark
        self.path = path
        self.df = self._load()

    def _load(self) -> DataFrame:
        return (
            self.spark.read
            .format("binaryFile")
            .load(self.path)
        )

    def parse(self) -> "PDFLoader":
        self.df = self.df.withColumn(
            "parsed",
            expr("ai_parse_document(content)")
        )
        self.df = self.df.withColumn("json_string", expr("to_json(parsed)"))
        return self

    def get_df(self) -> DataFrame:
        return self.df
    
    def get_json(self):
        data = self.df.select("json_string").toPandas()
        jsons = data['json_string'].tolist()

        return jsons