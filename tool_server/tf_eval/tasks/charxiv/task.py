
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import os
from datasets import load_dataset
logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)


def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    dataset = load_dataset(dataset_path)
    valset = dataset["validation"]
    testset = dataset["test"]
    
    val_metadata = []
    
    for idx,item in enumerate(valset):
        pass
        
    
    raw_data = []
    for k,v in load_json_file(dataset_path).items():
        raw_data.append(v)
        
    meta_data = []
    for idx,item in enumerate(raw_data):
        figure_id = item["figure_id"]
        item_id = f"charxiv_{figure_id}_{idx}"
        image_path = os.path.join(image_dir_path, f"{figure_id}.jpg")
        text = item["query"]

        data_item = dict(idx=item_id, image_path=image_path, text=text, **item)
        meta_data.append(data_item)
    meta_data = meta_data[:num_samples]
    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data



def evaluate_function(results, meta_data):
    """
    Evaluate the model predictions based on the provided results and metadata.

    Args:
        results (list): A list of model predictions, each corresponding to a question.
        meta_data (list): A list of metadata dictionaries, each containing:
                          - 'question' (str): The question asked.
                          - 'ground_truth' (str): The correct answer.
                          - 'type' (str): Type of question ('descriptive' or 'reasoning').

    Returns:
        dict: A dictionary containing evaluation metrics like accuracy (ACC).
    """
    if not results or not meta_data or len(results) != len(meta_data):
        raise ValueError("Results and meta_data must be non-empty and of the same length.")

    correct_count = 0
    total_count = len(results)
    type_correct = {'descriptive': 0, 'reasoning': 0}
    type_total = {'descriptive': 0, 'reasoning': 0}

    # Evaluate each result
    for result, meta in zip(results, meta_data):
        question_type = meta.get('type', 'unknown')
        ground_truth = meta.get('ground_truth', '').strip().lower()
        prediction = result.strip().lower()

        # Count totals by type
        if question_type in type_total:
            type_total[question_type] += 1

        # Check if the prediction matches the ground truth
        if prediction == ground_truth:
            correct_count += 1
            if question_type in type_correct:
                type_correct[question_type] += 1

    # Calculate overall accuracy
    overall_acc = correct_count / total_count if total_count > 0 else 0.0

    # Calculate accuracy by type
    acc_by_type = {
        q_type: type_correct[q_type] / type_total[q_type] if type_total[q_type] > 0 else 0.0
        for q_type in type_total
    }

    return {
        'overall_accuracy': overall_acc,
        'accuracy_by_type': acc_by_type,
        'total_questions': total_count,
        'correct_answers': correct_count,
    }