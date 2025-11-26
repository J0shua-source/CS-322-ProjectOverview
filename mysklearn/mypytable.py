from mysklearn import myutils

##############################################
# Programmer: Chris Wong
# Class: CptS 322-01, Fall 2025
# Assignment Project
# 12/1/25
# 
# 
# Description: This module defines the MyPyTable class, which represents a 2D table of data with various methods for data manipulation and analysis.
# It includes functionalities for loading/saving data from/to CSV files, handling missing values, finding duplicates, computing summary statistics, 
# and performing joins with other tables.
##############################################

import copy
import csv
import os

from  statistics import mean, median
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self, max_rows=50):
        """Prints the table in a nicely formatted grid structure.

        Parameters:
            max_rows (int): The maximum number of rows to display.
        """
        if not self.data:
            print("<empty table>")
            return
        
        if tabulate:
            print(tabulate(self.data[:max_rows], headers=self.column_names, tablefmt="grid"))
        else:
            # Fallback printer if tabulate is not installed
            # The fix is here: change map(str.self.column_names) to map(str, self.column_names)
            header = " | ".join(map(str, self.column_names))
            print(header)
            print("-" * len(header))
            for row in self.data[:max_rows]:
                print(" | ".join(map(str, row)))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        num_rows = len(self.data)
        num_cols = len(self.column_names)
        return num_rows, num_cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # resolve column index from name or integer
        if isinstance(col_identifier, int):
            col_index = col_identifier
        elif isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Invalid col_identifier: {col_identifier}")
            col_index = self.column_names.index(col_identifier)
        else:
            raise ValueError("col_identifier must be a string or integer")

        column_data = []
        for row in self.data:
            # protect against ragged rows
            if col_index < 0 or col_index >= len(row):
                raise IndexError(f"Column index out of range: {col_index}")
            value = row[col_index]
            if (value == "NA" or value is None) and not include_missing_values:
                # skip missing values when requested
                continue
            column_data.append(value)
        return column_data
    

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_value = float(self.data[i][j])
                    self.data[i][j] = numeric_value
                except (ValueError, TypeError):
                    # Leave the value as-is if it cannot be converted
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Sort in reverse to avoid index shifting issues while deleting
        for index in sorted(set(row_indexes_to_drop), reverse=True):
            if 0 <= index < len(self.data):
                del self.data[index]

    def load_from_file(self, filename, delimiter=None, has_header=True):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline='', encoding='utf-8') as infile:
            if delimiter is None:
                try:
                    dialect = csv.Sniffer().sniff(infile.read(1024))
                    infile.seek(0)
                    reader = csv.reader(infile, dialect)
                except csv.Error:
                    infile.seek(0)
                    reader = csv.reader(infile)
            else:
                reader = csv.reader(infile, delimiter=delimiter)

            if has_header:
                self.column_names = [name.strip() for name in next(reader)] 

            self.data = [[value.strip() for value in row] for row in reader if row]
        
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            if self.column_names:
                writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        indices = [self.column_names.index(name) for name in key_column_names]
        seen = set()
        duplicates = []
        for i, row in enumerate(self.data):
            key = tuple(row[index] for index in indices)
            if key in seen:
                duplicates.append(i)
            else:
                seen.add(key)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        # Create a new list excluding rows with "NA"
        self.data = [row for row in self.data if "NA" not in row]

    def get_columns_with_missing_values(table):
        count = 0
        missing_cols = []
        for col_idx, col_name in enumerate(table.column_names):
            for row in table.data:
                val = row[col_idx]
                if val is None or val == "" or (isinstance(val, str) and val.strip().lower() in ["na", "n/a", "?"]):
                    missing_cols.append(col_name)
                    count += 1
                    break
        return missing_cols


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_data = self.get_column(col_name, include_missing_values=False)
        numeric_values = [value for value in col_data if isinstance(value, (int, float))]
        
        if not numeric_values:
            return TypeError("No numeric data available to compute average.")
        
        avg_value = mean(numeric_values)
        col_index = self.column_names.index(col_name)
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg_value

    

    def clean_dataset(self, input_file, output_file, key_columns):
        """Loads a dataset, finds and removes duplicates, and saves the cleaned data.
        
        1. Loads data from input_file.
        2. Displays the number of instances.
        3. Finds and displays duplicates based on key_columns.
        4. If duplicates exist, removes them and saves the result to output_file.

        Parameters:
            input_file (str): The path to the input file.
            output_file (str): The path to save the cleaned output file.
            key_columns (list of str): Column names to use as keys for finding duplicates.
        """
        self.load_from_file(input_file)
        
        num_rows, _ = self.get_shape()
        print(f"\n--- Cleaning {os.path.basename(input_file)} ---")
        print(f"Number of instances: {num_rows}")

        dup_indexes = self.find_duplicates(key_columns)

        if dup_indexes:
            print(f"Found {len(dup_indexes)} duplicate(s) based on {key_columns}:")
            for index in dup_indexes:
                print(f"  - Row {index}: {self.data[index]}")
            
            self.drop_rows(dup_indexes)
            print(f"Removed {len(dup_indexes)} duplicate row(s).")

            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            self.save_to_file(output_file)
            print(f"Cleaned dataset written to: {output_file}")
        else:
            print("No duplicates found.")

   
    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_header = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []

        for name in col_names:
            col_data = self.get_column(name, include_missing_values=False)
            numeric_values = [value for value in col_data if isinstance(value, (int, float))]
            
            if not numeric_values:
                continue
            
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            mid_val = (min_val + max_val) / 2
            avg_val = mean(numeric_values)
            median_val = median(numeric_values)
            
            stats_data.append([name, min_val, max_val, mid_val, avg_val, median_val])
        
        return MyPyTable(stats_header, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        left_indices = [self.column_names.index(name) for name in key_column_names]
        right_indices = [other_table.column_names.index(name) for name in key_column_names]
        
        other_header_no_keys = [name for name in other_table.column_names if name not in key_column_names]
        new_header = self.column_names + other_header_no_keys

        right_table_index = {}
        for row in other_table.data:
            key = tuple(row[index] for index in right_indices)
            if key not in right_table_index:
                right_table_index[key] = []
            right_table_index[key].append(row)

        new_data = []
        for left_row in self.data:
            left_key = tuple(left_row[index] for index in left_indices)
            if left_key in right_table_index:
                for right_row in right_table_index[left_key]:
                    right_row_no_keys = [value for j, value in enumerate(right_row) if other_table.column_names[j] not in key_column_names]
                    new_data.append(left_row + right_row_no_keys)

        return MyPyTable(new_header, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        inner_join_table = self.perform_inner_join(other_table, key_column_names)
        new_data = copy.deepcopy(inner_join_table.data)
        new_header = copy.deepcopy(inner_join_table.column_names)

        left_indices = [self.column_names.index(name) for name in key_column_names]
        right_indices = [other_table.column_names.index(name) for name in key_column_names]

        right_keys = {tuple(row[index] for index in right_indices) for row in other_table.data}
        num_other_cols = len(other_table.column_names) - len(key_column_names)
        
        for left_row in self.data:
            left_key = tuple(left_row[index] for index in left_indices)
            if left_key not in right_keys:
                new_row = left_row + ["NA"] * num_other_cols
                new_data.append(new_row)
        
        left_keys = {tuple(row[index] for index in left_indices) for row in self.data}

        for right_row in other_table.data:
            right_key = tuple(right_row[index] for index in right_indices)
            if right_key not in left_keys:
                new_row = []
                for col in new_header:
                    if col in other_table.column_names:
                        new_row.append(right_row[other_table.column_names.index(col)])
                    else:
                        new_row.append("NA")
                new_data.append(new_row)

        return MyPyTable(new_header, new_data)

