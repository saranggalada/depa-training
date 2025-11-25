#!/usr/bin/env python3
"""
Oncology Data Query Tool
Loads all datasets from the data folder and allows PySpark SQL queries
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

class OncologyDataQuery:
    def __init__(self, data_path="/home/depa-train-dev/depa-training/scenarios/oncology/data"):
        self.spark = SparkSession.builder \
            .appName("OncologyDataQuery") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("ERROR")
        
        self.data_path = data_path
        self.tables = {}
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """Load all datasets from the data folder"""
        # Load CSV files
        csv_files = [
            ("cell_metadata", "/mnt/remote/cancer_institute/cell_metadata.csv"),
            ("sc_expr_matrix", "/mnt/remote/genomics_lab/sc_expr_matrix.csv"),
            ("drug_response", "/mnt/remote/pharmaceutical_company/drug_response.csv")
        ]
        
        for table_name, file_path in csv_files:
            full_path = file_path
            if os.path.exists(full_path):
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)
                df.createOrReplaceTempView(table_name)
                self.tables[table_name] = df
                print(f"Loaded {table_name}: {df.count()} rows")
        
        # Load JSON files
        json_files = [
            ("cell_markers", "/mnt/remote/computational_biology_lab/cell_markers.json"),
            ("drug_targets", "/mnt/remote/pharmaceutical_company/drug_targets.json")
        ]
        
        for table_name, file_path in json_files:
            full_path = file_path
            if os.path.exists(full_path):
                df = self.spark.read.option("multiline", "true").json(full_path)
                df.createOrReplaceTempView(table_name)
                self.tables[table_name] = df
                print(f"Loaded {table_name}: {df.count()} rows")
    
    def query(self, sql_query, output_path=None):
        """Execute SQL query and optionally save results"""
        try:
            result = self.spark.sql(sql_query)
            
            if output_path:
                result.write.mode("overwrite").option("header", "true").csv(output_path)
                print(f"Results saved to {output_path}")
            
            return result
        except Exception as e:
            print(f"Query error: {e}")
            return None
    
    def show_tables(self):
        """Show available tables"""
        print("Available tables:")
        for table_name, df in self.tables.items():
            print(f"  {table_name}: {df.count()} rows")
    
    def describe_table(self, table_name):
        """Show schema and sample data for a table"""
        if table_name in self.tables:
            print(f"\nSchema for {table_name}:")
            self.tables[table_name].printSchema()
            print(f"\nSample data from {table_name}:")
            self.tables[table_name].show(5, truncate=False)
        else:
            print(f"Table {table_name} not found")

    def save_results(self, result, output_path):
        """Save results to a CSV file"""
        result.write.mode("overwrite").option("header", "true").csv(output_path)
        print(f"Results saved to {output_path}")
    
    def close(self):
        """Close Spark session"""
        self.spark.stop()

def main():
    # Initialize the query tool
    query_tool = OncologyDataQuery()
    
    try:
        # Show available tables
        # query_tool.show_tables()
        
        # First, let's see what tables we have and their structure
        # query_tool.describe_table("sc_expr_matrix")
        # query_tool.describe_table("drug_targets")
        
        # Now that the matrix is transposed, it should have cells as rows and genes as columns
        print("Understanding transposed data structure...")
        
        # Get column information to see the new structure
        print("Getting column information...")
        columns_result = query_tool.query("DESCRIBE sc_expr_matrix")
        columns_result.show()        
            
        # Now create drug enrichment scores dynamically from drug targets
        print("Computing drug enrichment scores...")
        
        # With the transposed matrix, we can directly compute drug scores
        enrichment_result = query_tool.query("""
            SELECT 
                _c0 as cell_id,
                -- Venetoclax targets BCL2
                COALESCE(BCL2, 0) as Venetoclax,
                -- Azacitidine targets DNMT1 and IDH2
                COALESCE((DNMT1 + IDH2) / 2, 0) as Azacitidine,
                -- Dasatinib targets SRC, ABL1, KIT (using the first KIT column)
                COALESCE((SRC + ABL1 + KIT) / 3, 0) as Dasatinib,
                -- MK-2206 targets AKT1, AKT2, AKT3
                COALESCE((AKT1 + AKT2 + AKT3) / 3, 0) as MK_2206,
                -- Sorafenib targets FLT3, KIT (using the first KIT column)
                COALESCE((FLT3 + KIT) / 2, 0) as Sorafenib
            FROM sc_expr_matrix
            WHERE _c0 IS NOT NULL
            ORDER BY _c0
        """)
        
        if enrichment_result:
            enrichment_result.show(10)
            query_tool.save_results(enrichment_result, "/mnt/remote/features/drug_enrichment.csv")
    
    finally:
        query_tool.close()

if __name__ == "__main__":
    main()