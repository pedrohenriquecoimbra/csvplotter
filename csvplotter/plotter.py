import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import concurrent.futures

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

    @staticmethod
    def apply_style(style):
        """
        Apply the selected matplotlib style.
        """
        #style_path = f"Plotter/{self.style}.mplstyle"  # Look for style in the styles folder
        try: #if os.path.exists(style_path) or style_path in plt.style.available:
            plt.style.use(style)  # Apply the style
        except:
            print(
                f"Warning: Style '{style}' not found. Using default style.")
            plt.style.use("fivethirtyeight")  # Fallback style

    @staticmethod
    def _set_theme(theme, palette):
        """ Set theme for the plot """
        if theme == 'dark':
            sns.set_theme(style='darkgrid', palette=palette)
        else:
            sns.set_theme(style='whitegrid', palette=palette)

    @staticmethod
    def _get_figure_size(aspect):
        """ Return figure size based on aspect ratio """
        if isinstance(aspect, tuple):
            return aspect
        else:
            aspect_dict = {'small': (6, 4), 'medium': (
                8, 6), 'big': (10, 8), 'wide': (12, 6), 'very-wide': (24, 4)}
            return aspect_dict.get(aspect, (8, 6))

    @staticmethod
    def optimal_ylim(y_var, padding=0.05, percentiles=(5, 95)):
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
        y_data = y_var.dropna()

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

        if optimal_min == optimal_max:
            return None, None

        return optimal_min, optimal_max
    
    def windroseplot(self, config, save_as=None, data=None, opening=0.94, nsector=36, edgecolor='white', dpi=300):
        from windrose import WindroseAxes

        if data is None:  # Check explicitly for None
            data = self.data  # Use self.data if no data is provided

        fig = plt.figure(figsize=Plotter._get_figure_size(
            config.get('aspect', 'big')))  # Set the figure size
        ax = WindroseAxes.from_ax(fig=fig)  # Create a windrose axes
        # Plot the windrose
        ax.bar(data[config['x_var']], data[config['y_var']], normed=True,
               opening=opening, nsector=nsector, edgecolor=edgecolor)

        # Save the figure if requested
        if save_as:
            plt.savefig(save_as, dpi=dpi)
            plt.close()
        else:
            plt.show()
        return

    def plot_multiprocess_wrapper(self, args):
        config, kwargs = args
        return self.plot_wrapper(config, **kwargs)

    def plot_wrapper(self, config, *args, **kwargs):
        if config.get('kind', None) == 'windrose':
            self.windroseplot(config, *args, **kwargs)
        else:
            self.plot(config, *args, **kwargs)
        return
    
    def plot(self, config, save_as=None, data=None, dpi=None, legend_kwargs={}, **kwargs):
        """ Plot based on the configuration from CSV """
        if data is None:  # Check explicitly for None
            data = self.data  # Use self.data if no data is provided

        config.update(kwargs)

        # Set the y_var as a list if it is a string
        if isinstance(config['y_var'], str):
            config['y_var'] = [config['y_var']]

        # Only keep the columns that are present in the data
        config['x_var'] = config['x_var'] if config['x_var'] in data.columns else ''
        config['y_var'] = [y for y in config['y_var'] if y in data.columns]

        # Set the labels if not provided
        if not config.get('y_var_label', None):
            config['y_var_label'] = config['y_var']

        self.apply_style(self.style)
        
        #self._set_theme(config.get('theme', 'light'), config.get('palette', 'viridis'))
        
        plt.figure(figsize=Plotter._get_figure_size(
            config.get('aspect', 'wide')))
        
        if isinstance(config['y_var'], str):
            config['y_var'] = [config['y_var']]
            
        for y in config['y_var']:
            plot = sns.lineplot(data=data, x=config['x_var'], y=y,
                                hue=config['hue'], style=config['style'], size=config['size'])
        
        if config.get('x_label', None):
            plt.xlabel(config['x_label'])
        if config.get('y_label', None):
            plt.ylabel(config['y_label'])
        if config.get('title', None):
            plt.title(config['title'])
        if config.get('xlim', None):
            plt.xlim(config['xlim'])
        if config.get('ylim', None):
            plt.ylim(config['ylim'])
        else:
            plt.ylim(Plotter.optimal_ylim(data[config['y_var']]))
        
        # Add a legend if requested
        plt.legend(**legend_kwargs)
        # Save the figure if requested
        if save_as:
            plt.savefig(save_as, dpi=dpi)
            plt.close()
        else:
            plt.show()
        return
    
    def plot_from_csv(self, config, data=None,
                      theme='default', palette='viridis', aspect='big', 
                      save_folder=None, multi_process=False, **kwargs):
        """ Read CSV and plot all rows """
        # Read the configuration CSV file
        if isinstance(config, str) and os.path.exists(config):
            config_df = pd.read_csv(config).fillna(
                'None').replace({'None': None})
        else:
            config_df = pd.DataFrame(config)
        
        # Convert the y_var column to a list
        config_df['y_var'] = config_df['y_var'].replace(
            {None: "[]"}).apply(lambda x: ast.literal_eval(x) if '[' in x else x)

        # Prepare the list of jobs for multi-processing
        jobs = []
        for index, row in config_df.iterrows():
            config = row.to_dict()  # Convert row to dictionary
            file_name = config.pop('savn', None)
            config.update(dict(theme=theme, palette=palette, aspect=aspect))

            save_as = f"{save_folder}/{file_name}" if save_folder else file_name
            kwargs.update(dict(save_as=save_as, data=data))
            if multi_process:
                jobs.append((config, kwargs.copy()))
            else:
                self.plot_wrapper(config, **kwargs)

        if multi_process:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(self.plot_multiprocess_wrapper, jobs)
        return
