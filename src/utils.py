# src/utils.py

from typing import Any, Dict, List, Optional
from config import ConfigManager
import time
import psutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
import json
import numpy as np
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Timer Class
class Timer:
    """
    A flexible context manager and utility class for measuring execution time.

    Features:
    - Supports multiple timing units (seconds, milliseconds).
    - Allows labeling for easier identification in logs.
    - Tracks multiple timed sections when reused.
    """

    def __init__(self, unit: str = "seconds", label: Optional[str] = None):
        """
        Initialize the Timer with the specified unit and optional label.

        Args:
            unit (str): Time unit for logging ('seconds', 'milliseconds').
            label (Optional[str]): Optional label for the timer to add context to logs.

        Raises:
            ValueError: If the provided time unit is invalid.
        """
        self.unit = unit  # Unit for time measurement (seconds or milliseconds)
        self.label = label  # Optional label for the timer
        self.start_time: Optional[float] = None  # Start time of the timer
        self.end_time: Optional[float] = None  # End time of the timer
        self.elapsed_time: Optional[float] = None  # Elapsed time of the last session
        self.sections: List[float] = []  # List of elapsed times for multiple sections

        # Validate the unit argument
        if self.unit not in {"seconds", "milliseconds"}:
            raise ValueError("Invalid unit. Choose either 'seconds' or 'milliseconds'.")

    def __enter__(self) -> "Timer":
        """
        Start the timer when entering the context.

        Returns:
            Timer: The Timer instance to allow chaining operations or retrieving elapsed time.
        """
        self.start_time = time.time()  # Record the current time as the start time
        return self  # Return the instance for possible use within the context

    def __exit__(self, *_: Any) -> None:
        """
        Stop the timer and log the elapsed time when exiting the context.

        Args:
            *_: Standard arguments for exception handling in the context manager protocol,
                not used in this implementation.
        """
        self.stop()  # Stop the timer
        self.log_time()  # Log the elapsed time


    def stop(self) -> None:
        """
        Stop the timer manually and record the elapsed time.

        Raises:
            RuntimeError: If the timer was not started before calling stop.
        """
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Use __enter__() or call manually.")
        self.end_time = time.time()  # Record the current time as the end time
        self.elapsed_time = self.end_time - self.start_time  # Calculate elapsed time
        self.sections.append(self.elapsed_time)  # Store the elapsed time in the sections list

    def elapsed(self) -> float:
        """
        Get the elapsed time for the most recent timing session.

        Returns:
            float: The elapsed time in the specified unit (seconds or milliseconds).

        Raises:
            RuntimeError: If the timer has not been stopped yet.
        """
        if self.elapsed_time is None:
            raise RuntimeError("Timer has not been stopped yet. Call stop() first.")
        return self.elapsed_time * 1000 if self.unit == "milliseconds" else self.elapsed_time

    def log_time(self) -> None:
        """
        Log the elapsed time for the most recent session with the specified unit and label.

        Logs:
            - The elapsed time in seconds or milliseconds.
        """
        if self.elapsed_time is not None:
            elapsed = self.elapsed()  # Retrieve the elapsed time in the correct unit
            unit_label = "ms" if self.unit == "milliseconds" else "s"  # Determine unit label
            label = f" [{self.label}]" if self.label else ""  # Add label if provided
            logging.info(f"Execution time{label}: {elapsed:.2f} {unit_label}")

    def reset(self) -> None:
        """
        Reset the timer to start over, clearing all stored timing information.

        This is useful for reusing the timer in new contexts or sessions.
        """
        self.start_time = None  # Reset start time
        self.end_time = None  # Reset end time
        self.elapsed_time = None  # Reset elapsed time
        self.sections = []  # Clear all recorded sections

    def get_all_sections(self) -> List[float]:
        """
        Retrieve all recorded sections of elapsed times.

        Returns:
            List[float]: A list of elapsed times for all timed sections, converted to the specified unit.
        """
        return [sec * 1000 if self.unit == "milliseconds" else sec for sec in self.sections]


def log_memory_usage() -> None:
    """
    Logs the current memory usage of the Python process.

    This function uses the psutil library to fetch memory usage details of the current process.
    It logs the Resident Set Size (RSS) in megabytes (MB), which represents the physical memory used.
    """
    try:
        # Get the current process information
        process = psutil.Process()
        
        # Fetch memory usage details for the current process
        mem_info = process.memory_info()
        
        # Log memory usage in MB
        logging.info(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
    except psutil.Error as e:
        # Catch specific psutil-related errors
        logging.error(f"psutil encountered an error: {e}")
        sys.exit("Exiting due to failure in retrieving memory usage.")
    except Exception as e:
        # Catch any unexpected exceptions
        logging.error(f"Failed to retrieve memory usage: {e}")
        sys.exit("Exiting due to an unexpected failure in retrieving memory usage.")


def log_cpu_usage(interval: float = 1.0) -> None:
    """
    Logs the current CPU usage.

    This function uses the psutil library to fetch and log the CPU usage percentage and
    additional details such as the number of logical processors.

    Args:
        interval (float): The interval in seconds to calculate the CPU usage percentage.
                          Defaults to 1.0 (average CPU usage over 1 second).

    Raises:
        SystemExit: If there is an error while fetching CPU usage details.
    """
    try:
        # Validate the interval argument
        if not isinstance(interval, (int, float)) or interval <= 0:
            logging.error("The 'interval' parameter must be a positive number.")
            sys.exit("Exiting due to invalid 'interval' parameter in log_cpu_usage.")

        # Get CPU usage percentage
        cpu_usage = psutil.cpu_percent(interval=interval)

        # Get CPU count
        cpu_count = psutil.cpu_count(logical=True)

        # Log CPU usage details
        logging.info(f"CPU usage: {cpu_usage:.2f}%")
        logging.info(f"Logical processors: {cpu_count}")
    except psutil.Error as e:
        # Handle psutil-specific errors
        logging.error(f"psutil encountered an error while fetching CPU usage: {e}")
        sys.exit("Exiting due to failure in retrieving CPU usage details.")
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error in log_cpu_usage: {e}")
        sys.exit("Exiting due to an unexpected error in log_cpu_usage.")


def log_gpu_usage() -> None:
    """
    Logs GPU usage if TensorFlow detects GPU availability.

    This function checks for available GPUs using TensorFlow and logs memory statistics for each GPU.
    If no GPUs are detected, it logs a message indicating the system is running on a CPU.
    """
    try:
        # List all available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            # Iterate through each GPU and fetch its memory usage
            for gpu in gpus:
                gpu_name = gpu.name if hasattr(gpu, 'name') else "Unknown GPU"  # Retrieve GPU name
                try:
                    # Fetch memory information for the specific GPU
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    allocated = memory_info['current'] / (1024 ** 2)  # Convert bytes to MB
                    reserved = memory_info['peak'] / (1024 ** 2)  # Convert bytes to MB

                    # Log GPU memory usage
                    logging.info(f"GPU: {gpu_name}")
                    logging.info(f"  Memory allocated: {allocated:.2f} MB")
                    logging.info(f"  Memory reserved: {reserved:.2f} MB")
                except Exception as e:
                    # Log errors related to GPU memory retrieval
                    logging.warning(f"Failed to retrieve memory info for GPU {gpu_name}: {e}")
        else:
            # Log if no GPUs are detected
            logging.info("No GPUs detected. Running on CPU.")
    except tf.errors.UnknownError as e:
        # Catch TensorFlow-related errors
        logging.error(f"TensorFlow encountered an error: {e}")
        sys.exit("Exiting due to failure in retrieving GPU usage.")
    except Exception as e:
        # Catch any unexpected exceptions
        logging.error(f"Failed to retrieve GPU usage information: {e}")
        sys.exit("Exiting due to an unexpected failure in retrieving GPU usage.")


def plot_training_metrics(metrics: Dict[str, List[float]], save_path: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Plots training metrics like loss and accuracy over epochs.

    Args:
        metrics (Dict[str, List[float]]): A dictionary containing metrics (e.g., {"loss": [...], "accuracy": [...]}).
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Raises:
        SystemExit: If input validation fails or if saving the plot fails.
    """
    try:
        if not isinstance(metrics, dict) or not all(isinstance(values, list) for values in metrics.values()):
            logging.error("Metrics must be a dictionary where each key maps to a list of float values.")
            sys.exit("Exiting due to invalid metrics format in plot_training_metrics.")

        for metric, values in metrics.items():
            if not all(isinstance(value, (int, float)) for value in values):
                logging.error(f"All values for the metric '{metric}' must be integers or floats.")
                sys.exit(f"Exiting due to invalid values in metric '{metric}' in plot_training_metrics.")
            plt.plot(values, label=metric.capitalize())

        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Training Metrics")
        plt.legend()
        plt.grid(True)

        if save_path:
            try:
                plt.savefig(save_path)
                logging.info(f"Training metrics plot saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save the training metrics plot: {e}")
                sys.exit("Exiting due to failure in saving the plot in plot_training_metrics.")
        
        if show_plot:
            plt.show()

        # Clear the figure after saving or displaying
        plt.clf()

    except Exception as e:
        logging.error(f"Unexpected error in plot_training_metrics: {e}")
        sys.exit("Exiting due to an unexpected error in plot_training_metrics.")


def plot_data_distribution(data: List[int], class_names: List[str], save_path: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Plots the data distribution for the given classes.

    Args:
        data (List[int]): List of class labels (e.g., [0, 1, 0, 2]).
        class_names (List[str]): List of class names corresponding to the labels.
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Raises:
        SystemExit: If input validation fails or if saving the plot fails.
    """
    try:
        if not isinstance(data, list) or not all(isinstance(label, int) for label in data):
            logging.error("The 'data' parameter must be a list of integers representing class labels.")
            sys.exit("Exiting due to invalid 'data' parameter in plot_data_distribution.")
        
        if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
            logging.error("The 'class_names' parameter must be a list of strings.")
            sys.exit("Exiting due to invalid 'class_names' parameter in plot_data_distribution.")

        unique_labels = sorted(set(data))
        if len(unique_labels) > len(class_names):
            logging.error(f"The number of unique class labels ({len(unique_labels)}) exceeds the number of class names ({len(class_names)}).")
            sys.exit("Exiting due to mismatched 'data' and 'class_names' in plot_data_distribution.")

        label_to_name = {label: class_names[idx] for idx, label in enumerate(unique_labels)}
        logging.info(f"Label-to-class-name mapping: {label_to_name}")

        data_named = [label_to_name[label] for label in data]

        sns.countplot(x=data_named)
        plt.xticks(ticks=range(len(unique_labels)), labels=[label_to_name[label] for label in unique_labels])
        plt.title("Data Distribution")
        plt.xlabel("Classes")
        plt.ylabel("Count")

        if save_path:
            try:
                plt.savefig(save_path)
                logging.info(f"Data distribution plot saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save the data distribution plot: {e}")
                sys.exit("Exiting due to failure in saving the plot in plot_data_distribution.")
        
        if show_plot:
            plt.show()

        # Clear the figure after saving or displaying
        plt.clf()

    except Exception as e:
        logging.error(f"Unexpected error in plot_data_distribution: {e}")
        sys.exit("Exiting due to an unexpected error in plot_data_distribution.")


def plot_activation_map(image: np.ndarray, activation: np.ndarray, save_path: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Visualizes an activation map over an image.

    Args:
        image (np.ndarray): The original image as a NumPy array.
        activation (np.ndarray): The activation map as a NumPy array.
        save_path (Optional[str], optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.

    Raises:
        SystemExit: If input validation fails or if saving the plot fails.
    """
    try:
        if not isinstance(image, np.ndarray):
            logging.error("The 'image' parameter must be a NumPy array.")
            sys.exit("Exiting due to invalid 'image' parameter in plot_activation_map.")
        
        if not isinstance(activation, np.ndarray):
            logging.error("The 'activation' parameter must be a NumPy array.")
            sys.exit("Exiting due to invalid 'activation' parameter in plot_activation_map.")

        plt.imshow(image, cmap="gray")
        plt.imshow(activation, alpha=0.6, cmap="jet")
        plt.colorbar()
        plt.title("Activation Map")

        if save_path:
            try:
                plt.savefig(save_path)
                logging.info(f"Activation map plot saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save the activation map plot: {e}")
                sys.exit("Exiting due to failure in saving the plot in plot_activation_map.")
        
        if show_plot:
            plt.show()

        # Clear the figure after saving or displaying
        plt.clf()

    except Exception as e:
        logging.error(f"Unexpected error in plot_activation_map: {e}")
        sys.exit("Exiting due to an unexpected error in plot_activation_map.")


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Ensures consistent behavior across runs for TensorFlow, NumPy, and Python's random module.

    Args:
        seed (int): Random seed value to be used for reproducibility.

    Raises:
        ValueError: If the seed value is not a valid integer.
    """
    try:
        # Validate the seed
        if not isinstance(seed, int):
            logging.error("The seed must be an integer.")
            sys.exit("Exiting due to invalid seed value in set_random_seed.")

        # Set random seeds for reproducibility
        np.random.seed(seed)  # Set seed for NumPy
        random.seed(seed)  # Set seed for Python's random module
        tf.random.set_seed(seed)  # Set seed for TensorFlow
        logging.info(f"Random seed set to {seed}")
    except Exception as e:
        logging.error(f"Unexpected error in setting random seed: {e}")
        sys.exit("Exiting due to an unexpected error in set_random_seed.")


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Useful for saving model metadata, predictions, configurations, etc.

    Args:
        data (Dict[str, Any]): The data dictionary to save.
        file_path (str): The path where the JSON file will be saved.

    Raises:
        SystemExit: If the file cannot be saved due to unexpected errors.
    """
    try:
        # Validate input
        if not isinstance(data, dict):
            logging.error("The 'data' parameter must be a dictionary.")
            sys.exit("Exiting due to invalid 'data' parameter in save_json.")
        if not isinstance(file_path, str):
            logging.error("The 'file_path' parameter must be a string.")
            sys.exit("Exiting due to invalid 'file_path' parameter in save_json.")

        # Write the dictionary to a JSON file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Data successfully saved to JSON file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file at {file_path}: {e}")
        sys.exit("Exiting due to failure in saving JSON file in save_json.")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file to load.

    Returns:
        Dict[str, Any]: The loaded data as a dictionary.

    Raises:
        SystemExit: If the file cannot be loaded due to invalid file paths or unexpected errors.
    """
    try:
        # Validate input
        if not isinstance(file_path, str):
            logging.error("The 'file_path' parameter must be a string.")
            sys.exit("Exiting due to invalid 'file_path' parameter in load_json.")

        # Open and load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.info(f"Data successfully loaded from JSON file: {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"The file at {file_path} was not found.")
        sys.exit(f"Exiting due to missing file at {file_path} in load_json.")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON file at {file_path}: {e}")
        sys.exit(f"Exiting due to invalid JSON format in file at {file_path}.")
    except Exception as e:
        logging.error(f"Unexpected error in loading JSON file at {file_path}: {e}")
        sys.exit("Exiting due to an unexpected error in load_json.")


if __name__ == "__main__":

    # Instantiate the ConfigManager with the default or specified configuration file
    config_manager = ConfigManager()

    try:
        # Validate the configuration to ensure all sections are correct
        config_manager.validate_config()
    except Exception as e:
        # Log any validation errors that occur
        logging.error(f"Validation error: {e}")

    # Set Random Seed for Reproducibility
    logging.info("Setting random seed...")
    set_random_seed(seed=42)

    # Timer Usage: Measure execution time of a block of code
    with Timer(unit="milliseconds", label="Sample Task"):
        logging.info("Simulating a task that takes 2 seconds...")
        time.sleep(2)  # Simulate a task that takes 2 seconds

    # Log System Resource Usage
    logging.info("Logging memory usage...")
    log_memory_usage()

    logging.info("Logging CPU usage...")
    log_cpu_usage(interval=0.5)

    logging.info("Logging GPU usage...")
    log_gpu_usage()

    # Example Data for Plotting
    metrics = {
        "loss": [0.9, 0.7, 0.5, 0.3],
        "accuracy": [0.1, 0.4, 0.7, 0.9],
    }
    class_labels = [0, 1, 0, 2, 1, 2, 0]
    class_names = ["Class A", "Class B", "Class C"]
    example_image = np.random.rand(10, 10)  # Dummy grayscale image
    activation_map = np.random.rand(10, 10)  # Dummy activation map

    # Get Results Path
    results_path = config_manager.get_config_section("paths.results")

    # Plot Training Metrics
    training_metrics_path = f"{results_path}/training_metrics.png"
    logging.info(f"Plotting training metrics and saving to {training_metrics_path}...")
    plot_training_metrics(metrics=metrics, save_path=training_metrics_path, show_plot=False)

    # Plot Data Distribution
    data_distribution_path = f"{results_path}/data_distribution.png"
    logging.info(f"Plotting data distribution and saving to {data_distribution_path}...")
    plot_data_distribution(data=class_labels, class_names=class_names, save_path=data_distribution_path, show_plot=False)

    # Plot Activation Map
    activation_map_path = f"{results_path}/activation_map.png"
    logging.info(f"Plotting activation map and saving to {activation_map_path}...")
    plot_activation_map(image=example_image, activation=activation_map, save_path=activation_map_path, show_plot=False)

    # Save and Load JSON Example
    sample_data = {"model_name": "sample_model", "metrics": metrics}
    json_path = f"{results_path}/sample.json"

    logging.info(f"Saving JSON data to {json_path}...")
    save_json(data=sample_data, file_path=json_path)

    logging.info(f"Loading JSON data from {json_path}...")
    loaded_data = load_json(file_path=json_path)
    logging.info(f"Loaded JSON data: {loaded_data}")

    logging.info("Main script execution completed.")