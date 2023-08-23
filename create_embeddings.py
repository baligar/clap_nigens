import os
import pickle
import torch
import laion_clap
from src.laion_clap.clap_module.factory import create_model
from src.laion_clap.training.data import tokenizer, get_audio_features

model_for_text, model_cfg = create_model(
    amodel_name= "HTSAT-base",
    tmodel_name = "roberta",
    pretrained = "/code/CLAP/pretr_models/music_speech_audioset_epoch_15_esc_89.98.pt", 
    device = torch.device('cuda:0'))

# Define your directory path here
directory_path = './NIGENS_cmd_pkls/'

# List of list names
list_names = [
    "dog_bark_commands",
    "male_speech_commands",
    "door_knock_commands",
    "telephone_ring_commands",
    "alarm_sounds_commands",
    "crying_baby_commands",
    "crash_commands",
    "running_engine_commands",
    "burning_fire_commands",
    "footsteps_commands",
    "female_speech_commands",
    "female_scream_commands",
    "male_scream_commands",
    "ringing_phone_commands",
    "piano_commands"
]

# Dictionary to store the embeddings for each command
embeddings = {}

for list_name in list_names:
    # Load each command list from its .pkl file
    with open(os.path.join(directory_path, f"{list_name}.pkl"), "rb") as f:
        command_list = pickle.load(f)
    
    # Process each list
    text_data = command_list
    text_tokes = tokenizer(text_data)
    text_embed = model_for_text.get_text_embedding(text_tokes)
    
    # Store the embeddings in the dictionary with the list name as the key
    embeddings[list_name] = text_embed

# Now, embeddings dictionary contains the embeddings for each of the command lists
print("Embeddings created")
