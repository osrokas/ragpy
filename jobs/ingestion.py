# Databricks notebook source

# COMMAND ----------
import json
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# COMMAND ----------
config = dbutils.widgets.get("config")
config = json.loads(config)

# COMMAND ----------
catalog = config.get("catalog", "workspace")
schema = config.get("schema", "default")
src_dir = config.get("src_dir", "src_dir")
checkpoint_base = config.get("checkpoint_base", "checkpoint_base")
prepared_table = config.get("prepared_table", "prepared_table")

options = config.get("options", {})

# COMMAND ----------
if "cloudFiles.schemaLocation" not in options:
    schema_location = f"{checkpoint_base}/{schema}/{prepared_table}/_schema"
    options["cloudFiles.schemaLocation"] = schema_location

# COMMAND ----------
def read_autoloader(source_dir: str, target_table: str, catalog: str = "workspace", 
                    schema: str = "default", options: dict[str, Any] | None = None) -> DataFrame:
    """
    Reads data from a source directory using Databricks Autoloader.

    Parameters:
        - source_dir (str): The source directory to read data from.
        - target_table (str): The target table name.
        - catalog (str): The catalog name.
        - schema (str): The schema name.
        - options (dict): Additional options for the Autoloader.

    Returns:
        DataFrame: A DataFrame with the loaded data.

    Examples:
            options =  {"cloudFiles.format" : "csv",
                        "cloudFiles.schemaLocation" : schema_location,
                        "cloudFiles.inferColumnTypes" : "true",
                        "cloudFiles.schemaEvolutionMode" : "addNewColumns",
                        "header" : "true",
                        "delimiter" : ";",
                        "quote" : "\"",
                        "encoding" : "UTF-8"}

            read_autoloader(src_dir, target_table, catalog, schema, options)
    """
    df = (
        spark.readStream.format("cloudFiles")
            .options(**options)
            .load(source_dir)
            .select(
                "*",
                F.col("_metadata.file_path").alias("source_file_path"),
                F.current_timestamp().alias("load_timestamp"),
            )
        )  
    
    return (df.writeStream\
             .format("delta")\
             .outputMode("append")\
             .option("checkpointLocation", f"{options['cloudFiles.schemaLocation']}/_checkpoint")\
             .trigger(availableNow=True)\
             .table(f"{catalog}.{schema}.{target_table}")
             .awaitTermination())

# COMMAND ----------
read_autoloader(src_dir, prepared_table, catalog, schema, options)