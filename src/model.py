import os
import torch
import torch.nn as nn

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SAE(torch.nn.Module):
  def __init__(self):
    with_batch_normalization = True
    dropout_probability = 0

    super().__init__()
    self.encoder = torch.nn.Sequential(
      nn.Conv2d(in_channels=1  , out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.MaxPool2d(kernel_size=(2, 2)),
    )

    self.decoder = torch.nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), padding=2),
      nn.BatchNorm2d(128) if with_batch_normalization else nn.Identity(),
      nn.ReLU(),
      nn.Dropout2d(p = dropout_probability),
      nn.Upsample(scale_factor=(2, 2)),

      nn.Conv2d(in_channels=128, out_channels=1  , kernel_size=(5, 5), padding=2),
      nn.Sigmoid(),
    )

  def set_dropout_probability(self, dropout_probability = 0.2):
    for module in self.modules():
      if 'Dropout' in type(module).__name__:
        module.p = dropout_probability

  def enable_eval_dropout(self):
    for module in self.modules():
      if 'Dropout' in type(module).__name__:
        module.train()

  def forward(self, x):
    # Checks if input if batched or not. If it's not batched, add a dimension
    addDimension = x.dim() == 3

    if addDimension:
      x = torch.stack([x])

    x = self.encoder(x)
    x = self.decoder(x)

    # If we added a dimension, remove it for the loss function
    if addDimension:
      x = x.squeeze(0)

    return x
  
def getModelFileName(dataset_name: str, dropout_value: float, uses_redimension_vertical: bool, uses_redimension_horizontal: bool):
    sae_file = 'SAE'

    sae_file += f'_D{int(dropout_value * 10)}'

    if uses_redimension_vertical or uses_redimension_horizontal:
        sae_file += '_R'
        if uses_redimension_vertical:
            sae_file += 'V'
        if uses_redimension_horizontal:
            sae_file += 'H'

    return f'{sae_file}_{dataset_name}'


def getPredictionModel(dataset_name: str):
  model_path = f'models/SAE_{dataset_name}.pt'
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo de modelo {model_path} no existe.")
    
  model = torch.load(model_path, map_location=torch.device(DEVICE))
  model.to(DEVICE)
  return model


