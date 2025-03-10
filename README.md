# **Plotter Module**

A Python module to generate plots from a CSV configuration file using `seaborn`. This module allows you to create custom plots by reading plot configurations directly from a CSV file and automatically generating the plots according to the specified settings.

## Features
- **CSV-driven Plot Generation**: Read plot configurations from a CSV file and generate plots accordingly.
- **Customizable Plot Settings**: Control plot features such as axis labels, titles, themes, color palettes, and more.
- **Multiple Plot Sizes**: Choose from predefined aspect ratios (`small`, `medium`, `big`, `wide`).
- **Save Plots**: Optionally save plots as PNG, SVG, or PDF with custom resolutions.
- **Flexible Theme**: Select from light or dark themes.
- **Seaborn-powered**: Built on top of `seaborn`, making it easy to work with beautiful, statistical plots.

## Installation

To use this module, you need Python 3.x and the following libraries:
- `seaborn`
- `matplotlib`
- `pandas`

### Install via pip:

```bash
pip install seaborn matplotlib pandas
```

## Usage

### 1. **Importing the Module**:

```python
from plotter import Plotter
import pandas as pd
```

### 2. **Prepare Your Data**:
Ensure you have a `DataFrame` ready with the data to plot. For example:

```python
data = pd.DataFrame({
    'time': [1, 2, 3, 4, 5],
    'value': [10, 20, 15, 30, 25],
    'category': ['A', 'B', 'A', 'B', 'A']
})
```

### 3. **Prepare Your Configuration CSV File**:

Create a CSV file (`plot_config.csv`) with the following columns (or your own as needed):

| x_var | y_var | x_label | y_label | hue  | style | size | theme | palette | aspect | xlim | ylim | title |
|-------|-------|---------|---------|------|-------|------|-------|---------|--------|------|------|-------|
| time  | value | Time    | Value   | category |       |      | light | deep    | wide   | 0,35 | 0,40 | Sample Plot 1 |
| time  | value | Time    | Value   | category |       |      | dark  | pastel  | small  | 0,40 | 0,50 | Sample Plot 2 |

### 4. **Generate Plots from CSV**:

Call the `plot_from_csv` function to read the configuration and generate plots:

```python
plotter = Plotter(data)
plotter.plot_from_csv('plot_config.csv', data, save_folder="plots")
```

### Parameters:
- **csv_file**: Path to the CSV file containing plot configurations.
- **data**: DataFrame containing the data to plot.
- **save_folder** (optional): Folder where the plots will be saved. If not specified, plots will be shown without saving.
- **dpi** (optional): Resolution for saving plots (default is 300).

### 5. **Plot Customization**:
You can customize the plot appearance in the CSV:
- **x_var** and **y_var**: Column names for the X and Y axes.
- **x_label** and **y_label**: Labels for the X and Y axes.
- **hue**: Variable to use for color encoding.
- **style**: Variable to use for line style.
- **size**: Variable to use for size encoding.
- **theme**: Select between `'light'` or `'dark'` for plot background theme.
- **palette**: Choose color palette (e.g., `'deep'`, `'pastel'`, `'muted'`).
- **aspect**: Set plot aspect ratio (`'small'`, `'medium'`, `'big'`, `'wide'`).
- **xlim** and **ylim**: Set axis limits as tuples.
- **title**: Add a title to the plot.

### 6. **Example Code**:

```python
import pandas as pd
from plotter import Plotter

# Sample Data
data = pd.DataFrame({
    'time': [1, 2, 3, 4, 5],
    'value': [10, 20, 15, 30, 25],
    'category': ['A', 'B', 'A', 'B', 'A']
})

# Create the Plotter instance
plotter = Plotter(data)

# Plot the configurations from the CSV file
plotter.plot_from_csv('plot_config.csv', data, save_folder="plots")
```

### 7. **Saving the Plot**:
If you specify a folder in `save_folder`, the plots will be saved with the name `plot_1.png`, `plot_2.png`, etc. You can also specify custom file formats (e.g., PNG, SVG) and DPI.

## Example CSV (`plot_config.csv`):

```csv
x_var,y_var,x_label,y_label,hue,style,size,theme,palette,aspect,xlim,ylim,title
time,value,Time,Value,category,,medium,light,deep,wide,0,35,Sample Plot 1
time,value,Time,Value,category,,big,dark,pastel,small,0,40,Sample Plot 2
```

## Advanced Usage

You can extend the `Plotter` class to include more specific functionalities such as:
- Tooltips or zooming features (for future versions).
- Error handling for missing data or invalid configurations.

---

### Contributing

Feel free to open issues, create pull requests, or suggest improvements. This module is open for contributions and enhancements.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can modify and expand this template as your project evolves! It should give users a clear guide on how to get started with your module.