import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .utils import *

# Replace this with your actual code for data collection
results_dict = {
    ("LR: 0.001, Optimizer: Adam", "accuracy"): [0.85, 0.89, 0.92, 0.94, ...],
    ("LR: 0.001, Optimizer: Adam", "loss"): [0.2, 0.15, 0.1, 0.08, ...],
    ("LR: 0.01, Optimizer: SGD", "accuracy"): [0.88, 0.91, 0.93, 0.95, ...],
    ("LR: 0.01, Optimizer: SGD", "loss"): [0.18, 0.13, 0.09, 0.07, ...],
    ("LR: 0.1, Optimizer: Adam", "accuracy"): [0.82, 0.87, 0.91, 0.93, ...],
    ("LR: 0.1, Optimizer: Adam", "loss"): [0.25, 0.2, 0.15, 0.12, ...],
}
"""
server should have:
 #len = num of rounds
  self.accuracies
  self.losses
  
Client should have
  # len = num of epochs
  self.accuracies
  self.losses
  self.epsilons
  self.deltas
  
#common things : LR,SGD, Aggregation
  
"""


class AccuracyLossPlotter:
    """
    A utility class for plotting accuracy and loss metrics based on experiment results.

    Args:
        results_dict (dict): Dictionary containing experiment results organized by parameters and metrics.

    Attributes:
        results_dict (dict): Dictionary containing experiment results organized by parameters and metrics.
        parameters (list): List of unique parameters in the experiment results.
        metrics (list): List of unique metrics in the experiment results.
        iterations (range): Range of iterations (rounds or epochs) in the experiment.
    """

    def __init__(self, results_dict):
        """
        Initialize the AccuracyLossPlotter with experiment results.

        Args:
            results_dict (dict): Dictionary containing experiment results organized by parameters and metrics.
        """
        self.results_dict = results_dict
        self.parameters = list(
            set([param[0] for param in results_dict.keys()])
        )
        self.metrics = list(set([param[1] for param in results_dict.keys()]))
        self.iterations = range(1, len(list(results_dict.values())[0]) + 1)

    def plot_accuracy_loss(self):
        """
        Plot accuracy and loss metrics for different parameters.
        """

        plt.figure(figsize=(8, 6))

        for param in self.parameters:
            for metric in self.metrics:
                key = (param, metric)
                values = self.results_dict[key]
                plt.plot(
                    self.iterations,
                    values,
                    label=f"{param} ({metric})",
                    marker="o",
                    linestyle="-",
                )

            plt.xlabel("Rounds")
            plt.ylabel("Accuracy / Loss")
            plt.title("Accuracy and Loss by Parameters")
            plt.legend()
            plt.grid(True)
            plt.show()

    @staticmethod
    def plot_global_confusion_matrix(pipeline_name: str):
        """
        Plot a global confusion matrix based on pipeline results.

        Args:
            pipeline_name (str): Name of the pipeline.

        Returns:
            None
        """
        # Get the id of the pipeline by name
        pipeline_id = get_pipeline_from_name(pipeline_name)
        # get the confusion matrix pf the pipeline
        confusion_matrix = get_pipeline_confusion_matrix(pipeline_id)

        # Extracting confusion matrix values
        TP = confusion_matrix['TP']
        FP = confusion_matrix['FP']
        FN = confusion_matrix['FN']
        TN = confusion_matrix['TN']

        # Creating a matrix for visualization
        matrix = [[TN, FP],
                  [FN, TP]]

        # Plotting the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Global Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

        # Display the confusion matrix heatmap
        plt.show()

    @staticmethod
    def plot_confusion_Matrix_by_node(node_name: str, pipeline_name: str):
        """
        Plot a confusion matrix for a specific node in the pipeline.

        Args:
            node_name (str): Name of the node.
            pipeline_name (str): Name of the pipeline.

        Returns:
            None
        """

        # Get the id of the pipeline by name
        pipeline_id = get_pipeline_from_name(pipeline_name)
        # get the confusion matrix pf the pipeline
        confusion_matrix = get_node_confusion_matrix(
            pipeline_id, node_name=node_name)

        # Extracting confusion matrix values
        TP = confusion_matrix['TP']
        FP = confusion_matrix['FP']
        FN = confusion_matrix['FN']
        TN = confusion_matrix['TN']

        # Creating a matrix for visualization
        matrix = [[TN, FP],
                  [FN, TP]]

        # Plotting the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix of node: '+node_name)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()

        # Display the confusion matrix heatmap
        plt.show()
        return
    
    @staticmethod
    def plot_classification_report(pipeline_name: str):
        """
        Plot a comparison of classification report metrics between nodes.

        Args:
            pipeline_name (str): Name of the pipeline.

        Returns:
            None
        """

        colors = ['#FF5733', '#6A5ACD', '#3CB371', '#FFD700', '#FFA500', '#8A2BE2', '#00FFFF', '#FF00FF', '#A52A2A', '#00FF00']

        # Get the id of the pipeline by name
        pipeline_id = get_pipeline_from_name(pipeline_name)

        pipeline_results = get_pipeline_result(pipeline_id)

        nodesList = pipeline_results['nodename']
        classificationReports = []

        for index, node in enumerate(nodesList):
            classificationReports.append({
                'Accuracy': pipeline_results['accuracy'][index],
                'Sensitivity/Recall': pipeline_results['sensivity'][index],
                'PPV/Precision': pipeline_results['ppv'][index],
                'NPV': pipeline_results['npv'][index],
                'F1-score': pipeline_results['f1score'][index],
                'False positive rate': pipeline_results['fpr'][index],
                'True positive rate': pipeline_results['tpr'][index]
            })

        metric_labels = list(classificationReports[0].keys())  # Assuming both reports have the same keys

        # Set the positions of the bars on the x-axis
        x = np.arange(len(metric_labels))

        # Set the width of the bars
        width = 0.35

        plt.figure(figsize=(12, 6))

        for index, report in enumerate(classificationReports):
            metric = list(report.values())
            plt.bar(x + (index - len(nodesList) / 2) * width / len(nodesList), metric, width / len(nodesList),
                    label=nodesList[index], color=colors[index % len(colors)])

        # Adding labels, title, and legend
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Comparison of Classification Report Metrics between Nodes')
        plt.xticks(ticks=x, labels=metric_labels, rotation=45)
        plt.legend()

        # Show plot
        plt.tight_layout()
        plt.show()

        return 
