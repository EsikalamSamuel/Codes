Here's a brief overview of what this script does:

First I moved the 'dataset.csv' file into the home directory of the project.

I ran "pip install -r requirement.txt", then navigated to the main.py file and ran "python main.py".

The script creates before and after files and proceeds to generate visualizations, before the dataset is cleaned, stores it to 'before' and generates visualizations after cleaning and stores them as 'after'.

1. It defines a function `print_unique_count_of_values` to print the unique values and their counts for each column in the DataFrame.

2. It defines a function `replace_value` to replace specified values in the DataFrame with new values.

3. It defines a function `plot` to generate histograms for numeric columns and bar plots for categorical columns, saving them as image files.

4. It defines a function `examine_variable_dependency` to perform a chi-squared test for variable dependencies between each column and the 'embauche' column.

5. In the `analysis` function, it reads the "dataset.csv" file into a DataFrame, performs value replacements of '?', generates plots before and after replacements, examines variable dependencies, and finally saves the modified DataFrame to "processed.csv."

6. The script then calls the `analysis` function on the "dataset.csv" file.
