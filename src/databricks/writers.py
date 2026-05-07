from pyspark.sql import SparkSession

from src.models import ParsedDocument

class DatabricksRAGWriter():
    def __init__(self, spark: SparkSession, parsed_jsons: ParsedDocument):
        self.spark = spark
        self.parsed_jsons = parsed_jsons.to_dict()

    def write_table(self, table: str) -> None:
        df = self.spark.createDataFrame(self.parsed_jsons)
        df.write.mode('overwrite').option("overwriteSchema", "true").saveAsTable(table)