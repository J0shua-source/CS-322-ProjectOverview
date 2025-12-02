##############################################
# Programmer: Chris Wong
# Class: CptS 322-01, Fall 2025
# Programming Assignment #3
# 10/8/25
# 
# 
# Description: This module defines functions for visualizing data from a MyPyTable. 
# It includes functions to plot frequency diagrams, histograms, scatter plots, and box plots.
# These functions utilize matplotlib for generating the plots and handle various data types and edge cases.
##############################################

import matplotlib.pyplot as plt
import copy

def plot_frequency_diagram(table, column_name):
    """Generates and displays a frequency diagram (bar chart) for a given column.

    Args:
        table (MyPyTable): The table containing the data.
        column_name (str): The name of the column to plot.
    """
    try:
        column_data = table.get_column(column_name)
        
        frequencies = {}
        for value in column_data:
            frequencies[value] = frequencies.get(value, 0) + 1
            
        # Custom sorting key to handle mixed types (numbers, strings, ranges)
        def sort_key(item):
            key_val = item[0]
            # First, handle numeric types
            if isinstance(key_val, (int, float)):
                return (0, key_val)  # Group numbers first, then sort by value
            
            # Handle non-numeric types, converting to string
            key_str = str(key_val)
            # Check if the label is a range (e.g., "9.0--15.8")
            if '--' in key_str:
                try:
                    # Extract the first number to use for sorting
                    start_num = float(key_str.split('--')[0])
                    return (0, start_num) # Treat as a number for sorting
                except (ValueError, IndexError):
                    pass # Fall through to default string sorting
            
            # Group all other strings last, and sort them alphabetically
            return (1, key_str)

        sorted_items = sorted(frequencies.items(), key=sort_key)
        
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        # Convert all labels to string for plotting
        str_labels = [str(label) for label in labels]
        
        plt.figure(figsize=(10, 6))
        plt.bar(str_labels, counts, edgecolor='black')
        
        plt.title(f"Frequency Diagram of {column_name.title()}")
        plt.xlabel(column_name.title())
        plt.ylabel("Frequency (Count)")
        
        if len(labels) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print(f"Could not generate plot for '{column_name}' as it was not found in the table.")
    except Exception as e:
        print(f"An error occurred while plotting '{column_name}': {e}")

def plot_discretized_frequency_diagram(table, column_to_discretize, num_bins=5):
    """Discretizes a continuous attribute and plots its frequency diagram.

    Args:
        table (MyPyTable): The table containing the data.
        column_to_discretize (str): The name of the continuous column to bin.
        num_bins (int): The number of equal-width bins to create.
    """
    print(f"\n--- Discretizing '{column_to_discretize}' with {num_bins} bins ---")
    
    binned_table = copy.deepcopy(table)
    
    try:
        col_data = [val for val in binned_table.get_column(column_to_discretize) if isinstance(val, (int, float))]
    except ValueError:
        print(f"Column '{column_to_discretize}' not found.")
        return

    if not col_data:
        print(f"No numeric data found in column '{column_to_discretize}'.")
        return

    min_val = min(col_data)
    max_val = max(col_data)
    bin_width = (max_val - min_val) / num_bins
    
    bin_labels = []
    lower_bound = min_val
    for i in range(num_bins):
        upper_bound = lower_bound + bin_width
        label = f"{lower_bound:.1f}--{upper_bound:.1f}"
        if i == num_bins - 1:
            label = f"{lower_bound:.1f}--{max_val:.1f}" # Ensure last bin includes max
        bin_labels.append(label)
        lower_bound = upper_bound

    new_col_name = f"{column_to_discretize}_bins"
    binned_table.column_names.append(new_col_name)
    col_index = binned_table.column_names.index(column_to_discretize)

    for row in binned_table.data:
        value = row[col_index]
        bin_index = 0
        if isinstance(value, (int, float)) and value > min_val:
            bin_index = int((value - min_val) / bin_width)
            if bin_index >= num_bins:
                bin_index = num_bins - 1
        row.append(bin_labels[bin_index])

    plot_frequency_diagram(binned_table, new_col_name)


def plot_histogram(table, column_name):
    """Generates and displays a histogram for a given continuous attribute.

    Args:
        table (MyPyTable): The table containing the data.
        column_name (str): The name of the column to plot.
    """
    try:
        # Get the column data and filter for numeric types, ignoring Nones or non-numeric strings
        column_data = [val for val in table.get_column(column_name) if isinstance(val, (int, float))]
        
        if not column_data:
            print(f"No numeric data to plot for '{column_name}'.")
            return

        plt.figure(figsize=(10, 6))
        # Use plt.hist() with the default of 10 bins
        plt.hist(column_data, edgecolor='black') 
        
        plt.title(f"Histogram of {column_name.title()}")
        plt.xlabel(column_name.title())
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print(f"Could not generate histogram for '{column_name}' as it was not found in the table.")
    except Exception as e:
        print(f"An error occurred while plotting histogram for '{column_name}': {e}")


def plot_scatter(table, x_col_name, y_col_name):
    """Generates and displays a scatter plot for two continuous attributes.

    Args:
        table (MyPyTable): The table containing the data.
        x_col_name (str): The name of the column for the x-axis.
        y_col_name (str): The name of the column for the y-axis.
    """
    try:
        x_col_index = table.column_names.index(x_col_name)
        y_col_index = table.column_names.index(y_col_name)

        x_data = []
        y_data = []

        # Create pairs of (x, y) values, ensuring both are numeric
        for row in table.data:
            x_val = row[x_col_index]
            y_val = row[y_col_index]
            if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                x_data.append(x_val)
                y_data.append(y_val)
        
        if not x_data:
            print(f"No numeric data pairs to plot for '{x_col_name}' vs '{y_col_name}'.")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.5)
        
        plt.title(f"{y_col_name.title()} vs. {x_col_name.title()}")
        plt.xlabel(x_col_name.title())
        plt.ylabel(y_col_name.title())
        
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print(f"Could not generate scatter plot. One or both columns not found: '{x_col_name}', '{y_col_name}'.")
    except Exception as e:
        print(f"An error occurred while plotting scatter for '{x_col_name}' vs '{y_col_name}': {e}")


def plot_boxplot(table, x_col_name, y_col_name):
    """Generates and displays a box plot for a continuous attribute grouped by a categorical attribute.

    Args:
        table (MyPyTable): The table containing the data.
        x_col_name (str): The name of the categorical column for the x-axis.
        y_col_name (str): The name of the continuous column for the y-axis.
    """
    try:
        x_col_index = table.column_names.index(x_col_name)
        y_col_index = table.column_names.index(y_col_name)

        # Group y_values by the unique values in the x_column
        grouped_data = {}
        for row in table.data:
            x_val = row[x_col_index]
            y_val = row[y_col_index]

            # Ensure the y-value is numeric before adding
            if x_val is not None and isinstance(y_val, (int, float)):
                if x_val not in grouped_data:
                    grouped_data[x_val] = []
                grouped_data[x_val].append(y_val)

        if not grouped_data:
            print(f"No valid numeric data to plot for '{y_col_name}' grouped by '{x_col_name}'.")
            return

        # Sort the groups by the categorical label (e.g., model year)
        sorted_items = sorted(grouped_data.items(), key=lambda item: item[0])
        
        labels = [str(item[0]) for item in sorted_items]
        data_to_plot = [item[1] for item in sorted_items]

        plt.figure(figsize=(12, 7))
        plt.boxplot(data_to_plot)
        
        plt.title(f"Distribution of {y_col_name.title()} by {x_col_name.title()}")
        plt.xlabel(x_col_name.title())
        plt.ylabel(y_col_name.title())
        # Set the x-axis labels to be the categories
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
        
        plt.tight_layout()
        plt.show()

    except ValueError:
        print(f"Could not generate box plot. One or both columns not found: '{x_col_name}', '{y_col_name}'.")
    except Exception as e:
        print(f"An error occurred while creating the box plot: {e}")