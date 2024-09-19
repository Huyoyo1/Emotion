import pandas as pd
import pandas as pd
import numpy as np
import time
import random
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_dialogues(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dialogues = file.readlines()
    return [dialogue.strip() for dialogue in dialogues]

def load_emotions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [[int(emotion) for emotion in line.strip().split()] for line in lines]

def create_and_save_csv(dialogues_file, emotions_file, output_csv):
    dialogues = load_dialogues(dialogues_file)
    emotions = load_emotions(emotions_file)

    emotion_map = {
        0: 'no emotion',
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'sadness',
        6: 'surprise'
    }
    
    dialogue_ids = []
    utterances = []
    emotion_labels = []
    
    # Process each dialogue
    current_dialogue_id = 0
    for dialogue, emotion_list in zip(dialogues, emotions):
        utterance_list = dialogue.split('__eou__')
        utterance_list = [utt.strip() for utt in utterance_list if utt.strip()]
        
        for utterance, emotion in zip(utterance_list, emotion_list):
            dialogue_ids.append(current_dialogue_id)
            utterances.append(utterance)
            emotion_labels.append(emotion_map[emotion])
        
        current_dialogue_id += 1
    
    df = pd.DataFrame({
        'dialogue_id': dialogue_ids,
        'utterance': utterances,
        'emotion': emotion_labels
    })
    
    # Save DataFrame as CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV saved successfully as {output_csv}")


# Example usage
dialogues_file = 'Replace with your actual file path'  
emotions_file = 'Replace with your actual file path'    
output_csv = 'Replace with your actual file path' 

create_and_save_csv(dialogues_file, emotions_file, output_csv)


client = OpenAI(
    api_key="Replace with your API key",  
    base_url=" ",
)

# Function to create the context window
def create_context_window(df, index, window_size):
    dialogue_id = df.loc[index, "dialogue_id"]
    
    # Filter the dataframe to include only the rows with the same dialogue_id
    same_dialogue = df[df["dialogue_id"] == dialogue_id].reset_index(drop=True)
    
    # Find the local index within the same_dialogue subset
    current_index = same_dialogue.index[same_dialogue.index == index % len(same_dialogue)].tolist()[0]

    # Get the context (previous and next utterances within the window size)
    prev_context = same_dialogue.iloc[max(0, current_index - window_size) : current_index]
    next_context = same_dialogue.iloc[current_index + 1 : min(current_index + 1 + window_size, len(same_dialogue))]

    # Combine the contexts
    context = pd.concat(
        [
            pd.DataFrame([{"utterance": "null", "Speaker": "null"}] * (window_size - len(prev_context))),
            prev_context,
            next_context,
            pd.DataFrame([{"utterance": "null", "Speaker": "null"}] * (window_size - len(next_context))),
        ]
    )
    
    # Get the current utterance
    current_utterance = same_dialogue.iloc[current_index]
    
    return current_utterance, context

    
   


# Function to generate the text (emotion prediction)
def generate_text(system, user, model_name="gpt-4o", max_tokens=2500, temperature=0):

    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # Assuming the response is structured with a list of messages
            response_content = response.choices[0].message.content.strip()
            break
        except Exception as e:
            slt = random.random() * 0.6
            print(f"Request failed: {e}, Retrying after {slt:.3f}s delay")
            time.sleep(slt)

    return response_content

def evaluate_with_gpt4o(df, window_size, delay_seconds=1):
    responses = []
    for idx in tqdm(range(len(df))):
        # Create context window and extract the current utterance
        current_utterance, context = create_context_window(df, idx, window_size)

        context_str = "\n".join(
            [f"Speaker: {r['utterance']}" for _, r in context.iterrows()]
        )

        user_message = (
            f"Here is a conversation snippet including the current utterance:\n\n{context_str}\n\n"
            f"Current Utterance: {current_utterance['utterance']}\n\n"
            "What is the most likely emotion expressed in the current utterance? Choose from: no emotion, anger, disgust, fear, happiness, sadness, surprise. Provide only ONE emotion as the output. Don't return any other things!"
        )

        system_message = (
            "You are a highly efficient assistant tasked with evaluating the emotion expressed in a conversation snippet. "
            "You need to provide the most accurate emotional label for the main utterance in the context provided."
        )

        # Call the GPT-4o API with the prepared messages
        predicted_emotion = generate_text(system_message, user_message)
        responses.append(predicted_emotion)

        # time.sleep(delay_seconds)

    # Assign predictions to DataFrame and ensure all emotion labels are in lowercase for consistent comparison
    df["Predicted_Emotion"] = responses
    df["Predicted_Emotion"] = df["Predicted_Emotion"].astype(str).str.lower()
    df["emotion"] = df["emotion"].astype(str).str.lower()

    return df


# Function to calculate metrics
def calculate_metrics(y_true, y_pred, labels):
    # Replace NaNs with a placeholder (could be 'unknown' or similar if applicable)
    y_true = np.nan_to_num(y_true, nan=-1)
    y_pred = np.nan_to_num(y_pred, nan=-1)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        zero_division=1,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "overall": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "f1_macro": report["macro avg"]["f1-score"],
            "precision_weighted": report["weighted avg"]["precision"],
            "recall_weighted": report["weighted avg"]["recall"],
            "f1_weighted": report["weighted avg"]["f1-score"],
        },
        "by_class": {},
    }
    for idx, label in enumerate(labels):
        true_positives = cm[idx, idx]
        total_predicted_positives = np.sum(cm[:, idx])
        class_accuracy = (
            true_positives / total_predicted_positives
            if total_predicted_positives > 0
            else 0
        )  

        metrics["by_class"][label] = {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1_score": report[label]["f1-score"],
            "accuracy": class_accuracy,
            "support": report[label]["support"],
        }

    return metrics

# Main Execution
# Load your dataset
file_path = 'Replace with your actual file path'  
df = pd.read_csv(file_path)


# Example window size
window_size = 3

# Evaluate the predictions
df_with_predictions = evaluate_with_gpt4o(df, window_size)

# Calculate the metrics
labels = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
metrics = calculate_metrics(df_with_predictions["emotion"], df_with_predictions["Predicted_Emotion"], labels)

print("Overall Metrics:")
print(f"Accuracy: {metrics['overall']['accuracy']}")
print(f"Macro Precision: {metrics['overall']['precision_macro']}")
print(f"Macro Recall: {metrics['overall']['recall_macro']}")
print(f"Macro F1 Score: {metrics['overall']['f1_macro']}")
print(f"Weighted Precision: {metrics['overall']['precision_weighted']}")
print(f"Weighted Recall: {metrics['overall']['recall_weighted']}")
print(f"Weighted F1 Score: {metrics['overall']['f1_weighted']}")

print("\nMetrics by Class:")
for label, class_metrics in metrics['by_class'].items():
    print(f"\nClass: {label}")
    print(f"Precision: {class_metrics['precision']}")
    print(f"Recall: {class_metrics['recall']}")
    print(f"F1 Score: {class_metrics['f1_score']}")
    print(f"Accuracy: {class_metrics['accuracy']}")
    print(f"Support: {class_metrics['support']}")

import json
with open('Dia_results_metrics_3.json', 'w') as f:
    json.dump(metrics, f, indent=4)

output_file_path = 'Dia_prediction_results_3.csv'
df_with_predictions.to_csv(output_file_path, index=False)
print(f"Results have been saved to {output_file_path}.")

    
