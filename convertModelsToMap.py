from src.model import SAE, getPredictionModel, DEVICE
from src.defines import DATASETS
import os
import torch
import torch.nn as nn

def convert_model_to_weights_only(dataset_name: str):
    model_path = f'models/SAE_{dataset_name}.pt'
    
    # Verifica si el archivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo de modelo {model_path} no existe.")
    
    # Carga el modelo completo
    model: SAE = torch.load(model_path, map_location=torch.device(DEVICE))
    model.to(DEVICE)
    
    # Guarda solo los pesos del modelo en el mismo archivo o en uno nuevo
    torch.save(model.state_dict(), model_path)
    print(f'Model {model_path} converted to weights only and saved.\n')

def reload_and_verify_model(dataset_name: str):
    model_path = f'models/SAE_{dataset_name}.pt'
    
    # Crea una nueva instancia del modelo y carga los pesos
    model = SAE()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(DEVICE)
    
    print(f'Model {model_path} reloaded successfully.\n')

if __name__ == '__main__':
    # Convertir todos los modelos en la lista DATASETS
    # for dataset_name in DATASETS:
    #     convert_model_to_weights_only(dataset_name)
    
    # Verificar que los modelos fueron cargados correctamente después de la conversión
    for dataset_name in DATASETS:
        reload_and_verify_model(dataset_name)
    