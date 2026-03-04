import os
import ast
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import concurrent.futures
import datetime
import argparse
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

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
    def validate_data_type(data):
        """
        Check data
        allowed types are str to netcdf, dict or pandas.DataFrame 
        """
        assert (isinstance(data, (str, pd.DataFrame))), f'Data format not accepted ({type(data)}).'
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            try:
                data = pd.read_csv(data)
            except:
                pass
        return None

    @staticmethod
    def apply_style(style):
        """
        Apply the selected matplotlib style.
        """
        #style_path = f"Plotter/{self.style}.mplstyle"  # Look for style in the styles folder
        try: #if os.path.exists(style_path) or style_path in plt.style.available:
            if style is not None:
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
        y_data = y_var#.dropna(how='all')

        # Calculate the desired percentiles to avoid extreme values
        lower_percentile = np.nanpercentile(y_data, percentiles[0])
        upper_percentile = np.nanpercentile(y_data, percentiles[1])

        # Calculate the range based on the percentiles
        range_span = upper_percentile - lower_percentile

        # Calculate the padding amount based on the range
        pad_amount = range_span * padding

        # Define the optimal y-limits
        optimal_min = lower_percentile - pad_amount
        optimal_max = upper_percentile + pad_amount

        # Ensure that the limits are not NaN or infinite
        optimal_min = None if (pd.isna(optimal_min) or np.isinf(optimal_min)) else optimal_min
        optimal_max = None if (pd.isna(optimal_max) or np.isinf(optimal_max)) else optimal_max

        # Ensure that the limits are not equal
        if optimal_min == optimal_max:
            return None, None

        return optimal_min, optimal_max

    @staticmethod
    def literal_eval(config):
        """ Evaluate literal string as Python code """
        config['y_var'] = config['y_var'].replace(
            {None: "[]"}).apply(lambda x: ast.literal_eval(x) if '[' in x else x)
        
        for p in ['xlim', 'ylim']:
            if p in config:
                config[p] = config[p].replace({None: "''"}).apply(
                    ast.literal_eval).replace({'': None})
        
        for p in ['y_var_label', 'origin_file']:
            if p in config:
                config[p] = config[p].replace({None: "[]"}).apply(
                    lambda x: ast.literal_eval(x) if '[' in x else x)
        return config

    @staticmethod
    def try_date_ticks(data, var, axis=0):
        if axis == 0:
            ticks = plt.xticks
        elif axis == 1:
            ticks = plt.yticks
        try:
            x_0000 = data[var][data[var].dt.time == datetime.time(0, 0)]
            if len(x_0000) > 3:
                if len(data[var].dt.month.unique()) > 3:
                    ticks([list(x_0000)[int(i)] for i in np.linspace(0, len(x_0000)-1, 5)],
                        [list(x_0000)[int(i)].strftime("%d/%m/%Y") for i in np.linspace(0, len(x_0000)-1, 5)])
                else:
                    ticks([list(x_0000)[int(i)] for i in np.linspace(0, len(x_0000)-1, min(len(x_0000), 5))],
                        [list(x_0000)[int(i)].strftime("%d/%m") for i in np.linspace(0, len(x_0000)-1, min(len(x_0000), 5))])
            else:
                # check if data_fc[x_var] is datetime type
                ticks([list(data[var])[int(i)] for i in np.linspace(0, len(data)-1, 5)],
                    [list(data[var])[int(i)].strftime("%d/%m %H:%M") for i in np.linspace(0, len(data)-1, 5)])
        except AttributeError:
            pass
        except Exception as e:
            print(e)
        return

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

    @staticmethod
    def __read_config__(config):
        """ Read CSV and plot all rows """
        # Read the configuration CSV file
        if isinstance(config, str) and os.path.exists(config):
            config = pd.read_csv(config).fillna(
                'None').replace({float('nan'): None, np.nan: None, 'None': None})
        else:
            config = pd.DataFrame(config)

        # Convert columns to Python objects
        config = Plotter.literal_eval(config)
        return config

    def plot_multiprocess_wrapper(self, args):
        config, kwargs = args
        return self.plot_wrapper(config, **kwargs)

    def plot_wrapper(self, config, error=None, *args, **kwargs):
        try:
            if config.get('kind', None) == 'windrose':
                return self.windroseplot(config, *args, **kwargs)
            else:
                return self.plot(config, *args, **kwargs)
        except Exception as e:
            if error == 'ignore':
                pass
            elif error == 'raise':
                raise e
            else:
                print(f"Error: {e}")
        return
    
    def plot(self, config: dict, save_as: str = None, data: pd.DataFrame = None, 
             dpi: int = None, legend_kwargs: dict = {}, **kwargs):
        """ Plot based on the configuration from CSV """
        config.update(kwargs)

        if data is None:  # Check explicitly for None
            data = self.data  # Use self.data if no data is provided
        regex = config.get('regex', False)

        # Set the y_var as a list if it is a string
        if isinstance(config['y_var'], str):
            config['y_var'] = [config['y_var']]

        # Only keep the columns that are present in the data
        config['x_var'] = config['x_var'] if config['x_var'] in data.columns else ''
        
        if regex:
            config['y_var'] = [c for y in config['y_var'] for c in data.filter(
                regex="(?i)" + y).columns.tolist()]
        else:
            config['y_var'] = [y for y in config['y_var'] if y in data.columns]
        
        # Set the labels if not provided
        if regex or not config.get('y_var_label', None):
            config['y_var_label'] = config['y_var']
        config['y_var_label'] = [l if l else v for v, l in zip(config['y_var'], config['y_var_label'])]

        # Apply the selected style
        self.apply_style(self.style)
        
        #self._set_theme(config.get('theme', 'light'), config.get('palette', 'viridis'))
        
        fig = plt.figure(figsize=Plotter._get_figure_size(
            config.get('aspect', 'wide')))

        if sum([(config[v] not in data.columns) for v in ['hue', 'style', 'size'] if config.get(v, None) is not None]) or \
                len(config['x_var'])==0 or len(config['y_var'])==0:
            # print("Error: One or more of the hue, style, or size variables are not in the data.")
            plt.close()
            return
        
        # Plot the data
        if config['x_var'] not in config['y_var']:
            data_m = data.melt(id_vars=[config['x_var']] + [config[k] for k in ['hue', 'style', 'size'] if config[k]],
                            value_vars=config['y_var'], var_name='variable', value_name='y_val')
            data_m['variable'] = data_m['variable'].replace(
                config['y_var'], config['y_var_label'])
            sns.lineplot(data=data_m, x=config['x_var'], y='y_val', hue='variable', **{k: config[k] for k in [
                        'style', 'size'] if config[k]})
        else:
            for y, l in list(zip(config['y_var'], config['y_var_label'])):
                plot_kwargs = {k: config[k] for k in [
                    'hue', 'style', 'size'] if config[k]} or {'label': l}
                sns.lineplot(data=data, x=config['x_var'], y=y, **plot_kwargs)
        
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
        
        # Try to set date ticks
        self.try_date_ticks(data, config['x_var'], axis=0)

        # Add a legend if requested
        plt.legend(**legend_kwargs)

        # Adjust the padding between and around subplots
        plt.tight_layout()

        # Save the figure if requested
        if save_as:
            plt.savefig(save_as, dpi=dpi)
        else:
            # Save the plot to a BytesIO buffer
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)  # Rewind the buffer

            # Load the buffer into a PIL Image
            pil_image = Image.open(buffer)
            
            plt.show()
            
            # Close the plot to avoid memory leaks
            plt.close()

            return pil_image
        
        # Close the plot to avoid memory leaks
        plt.close()
        return

    def serial_plot(self, config, data=None,
                    theme='default', palette='viridis', aspect='big',
                    multi_process=False, **kwargs):
        if isinstance(config, pd.DataFrame):
            config = list(config.T.to_dict().values())
        elif isinstance(config, pd.Series):
            config = [config.to_dict()]
        elif isinstance(config, dict):
            config = [config]
        
        # Prepare the list of jobs for multi-processing
        jobs = []
        results = []
        for row in config:
            cfg = dict(theme=theme, palette=palette, aspect=aspect)
            cfg.update(row)  # Convert row to dictionary
            
            kwargs.update(dict(save_as=cfg.pop('save_as', None), data=data))
            if multi_process:
                jobs.append((cfg, kwargs.copy()))
            else:
                results.append(self.plot_wrapper(cfg, **kwargs))

        if multi_process:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(self.plot_multiprocess_wrapper, jobs))
        return results
    
    def plot_from_csv(self, config, **kwargs):
        """ Read CSV and plot all rows """
        # Read the configuration CSV file
        config = Plotter.__read_config__(config)
        config = self.serial_plot(config, **kwargs)
        return


def __custom_params__(unknown_args):
    def convert_to_number(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    # Print or process unknown arguments
    custom_params = {}
    i = 0
    while i < len(unknown_args):
        logger.debug(f"Custom/unknown arguments: {unknown_args}")
        # You can store or process these as needed, e.g.:
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            # Check if the next argument is a value (not starting with '--')
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                custom_params[key] = convert_to_number(unknown_args[i + 1])
                i += 1  # Skip the next argument as it's the value
            else:
                custom_params[key] = True  # Flag argument
        else:
            # Handle values for the previous key if needed
            custom_params[key].append(convert_to_number(arg))
            pass
        i += 1
    return custom_params


def main():
    parser = argparse.ArgumentParser(description="Run FrameworkAPI workflows.")
    parser.add_argument("data",
                        help="pandas DataFrame containing the data to plot.")
    parser.add_argument("--style", default="default",
                        help="Name of the matplotlib style to use (corresponds to a .mplstyle file).")
    parser.add_argument("--config", required=True,
                        help="Path to CSV configuration file.")
    parser.add_argument("--aspect", default='big',
                        help="Aspect ratio (tuple or string, e.g.: (4, 3), 'small', 'big', 'wide', ...")
    parser.add_argument("--verbosity", default="INFO",
                        help="Logging level (e.g., DEBUG, INFO, WARNING)")
    # Parse known arguments and capture the rest
    args, unknown_args = parser.parse_known_args()

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if args.verbosity.upper() not in valid_levels:
        logger.error(
            f"Invalid verbosity level. Choose from: {valid_levels}")
        args.verbosity = "0"
    logging.basicConfig(level=args.verbosity.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    custom_params = __custom_params__(unknown_args)

    config = Plotter.__read_config__(args.config)
    Plotter(args.data, args.style).serial_plot(
            config, aspect=args.aspect, **custom_params)
    
    return


if __name__ == "__main__":
    main()
