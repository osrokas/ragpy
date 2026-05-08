import argparse

from pyspark.sql import SparkSession

from ragpy_modules.loaders import PDFLoader

def main(input_path: str, output_path: str):
    spark = SparkSession.builder.appName("PDFLoaderExample").getOrCreate()

    loader = PDFLoader(spark=spark, path=input_path)
    loader.parse()
    jsons = loader.get_json()

    # Save json to volume
    with open(output_path, "w") as f:
        for json in jsons:
            f.write(json + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load PDFs, parse them, and save the parsed JSON.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input PDFs.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the parsed JSON.")
    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path)