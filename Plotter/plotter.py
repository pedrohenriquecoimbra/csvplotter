import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Plotter:
    def __init__(self, data, style="default"):
        """
        Initialize the plotter.

        Parameters:
        - data: pandas DataFrame containing the data to plot.
        - style: Name of the matplotlib style to use (corresponds to a .mplstyle file).
        """
        self.data = data
        self.style = style
        self.apply_style()
    
    def apply_style(self):
        """
        Apply the selected matplotlib style.
        """
        #style_path = f"Plotter/{self.style}.mplstyle"  # Look for style in the styles folder
        try: #if os.path.exists(style_path) or style_path in plt.style.available:
            plt.style.use(self.style)  # Apply the style
        except:
            print(
                f"Warning: Style '{self.style}' not found. Using default style.")
            plt.style.use("fivethirtyeight")  # Fallback style
    
    def _set_theme(self, theme, palette):
        """ Set theme for the plot """
        if theme == 'dark':
            sns.set_theme(style='darkgrid', palette=palette)
        else:
            sns.set_theme(style='whitegrid', palette=palette)

    def _get_figure_size(self, aspect):
        """ Return figure size based on aspect ratio """
        aspect_dict = {'small': (6, 4), 'medium': (
            8, 6), 'big': (10, 8), 'wide': (12, 6)}
        return aspect_dict.get(aspect, (8, 6))

    def optimal_ylim(self, y_var, padding=0.05, percentiles=(5, 95)):
        """
        Calculate optimal y-limits to avoid aberrant values and outliers.

        Parameters:
        - y_var: The column name in the dataframe for which to calculate the limits.
        - padding: Padding percentage to add to the limits to ensure data visibility (default 5%).
        - percentiles: Tuple of percentiles to consider for calculating the limits (default (5, 95)).

        Returns:
        - A tuple of (min, max) values for y-limits.
        """
        # Extract the relevant data for y_var
        y_data = self.data[y_var].dropna()

        # Calculate the desired percentiles to avoid extreme values
        lower_percentile = np.percentile(y_data, percentiles[0])
        upper_percentile = np.percentile(y_data, percentiles[1])

        # Calculate the range based on the percentiles
        range_span = upper_percentile - lower_percentile

        # Calculate the padding amount based on the range
        pad_amount = range_span * padding

        # Define the optimal y-limits
        optimal_min = lower_percentile - pad_amount
        optimal_max = upper_percentile + pad_amount

        return optimal_min, optimal_max
    
    def plot(self, config, data, save_as=None, dpi=300):
        """ Plot based on the configuration from CSV """
        #self._set_theme(config.get('theme', 'light'), config.get('palette', 'viridis'))

        plt.figure(figsize=self._get_figure_size(config.get('aspect', 'big')))

        plot = sns.lineplot(data=data, x=config['x_var'], y=config['y_var'],
                            hue=config['hue'], style=config['style'], size=config['size'])

        if config['x_label']:
            plt.xlabel(config['x_label'])
        if config['y_label']:
            plt.ylabel(config['y_label'])
        if config['title']:
            plt.title(config['title'])
        if config['xlim']:
            plt.xlim(config['xlim'])
        if config['ylim']:
            plt.ylim(config['ylim'])
        else:
            plt.ylim(self.optimal_ylim(config['y_var']))

        # Save the figure if requested
        if save_as:
            plt.savefig(save_as, dpi=dpi)
            plt.close()
        else:
            plt.show()
        return

    def plot_from_csv(self, csv_file, data, 
                      theme='default', palette='viridis', aspect='big', 
                      save_folder=None):
        """ Read CSV and plot all rows """
        # Read the configuration CSV file
        config_df = pd.read_csv(csv_file).fillna('None').replace({'None': None})

        for index, row in config_df.iterrows():
            config = row.to_dict()  # Convert row to dictionary
            file_name = config.pop('savn', None)
            config.update(dict(theme=theme, palette=palette, aspect=aspect))

            save_as = f"{save_folder}/{file_name}" if save_folder else file_name
            self.plot(config, data, save_as=save_as)
        return
