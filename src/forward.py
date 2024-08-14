from torch import max as TMax, no_grad
import torchvision.transforms as T
from PIL import  Image as PILImage

from src.defines import PredictionsCombinationType, ForwardParameters, BBOX_REDIMENSIONED_RECOVER, SAE_IMAGE_SIZE
from src.model import DEVICE, getPredictionModel, SAE
from src.imageProcess import getConnectedComponents
from src.boxesManipulation import resize_box, normalize_box


def get_transform() -> T.Compose:
    t = [
        T.ToTensor(),
        T.Grayscale(),
        ]

    return T.Compose(t)

def forwardToModel(model: SAE, image, times_pass_model: int, type_combination: PredictionsCombinationType):
    if type_combination == PredictionsCombinationType.MEAN:
        result = model(image.to(DEVICE))
        for i in range(1, times_pass_model):
            print(f'\r\tForward passing with mean result {i+1}', end='')
            result += model(image.to(DEVICE))
        result /= times_pass_model
        print()
        return result.cpu()

    elif type_combination == PredictionsCombinationType.MAX:
        result = model(image.to(DEVICE))
        for i in range(1, times_pass_model):
            print(f'\r\tForward passing with max result {i+1}', end='')
            result = TMax(result, model(image.to(DEVICE)))
        print()
        return result.cpu()

    else:
        return model(image.to(DEVICE)).cpu()

def TFMForward(
    dataset_name: str, image: PILImage, forwardParameters: ForwardParameters
    ):
    """Forward de una imagen para obtener los bounding boxes

    Args:
        dataset_name (string): Nombre del dataset
        image (Image): Imagen a hacer forward
        uses_redimension_vertical (bool): Si el modelo usa redimensión vertical
        uses_redimension_horizontal (bool): Si el modelo usa redimensión horizontal
        bin_umbral (float): Umbral de binarización
        val_dropout (float): Dropout a aplicar en prediccion
        times_pass_model (int): Cantidad de predicciones a combinar en prediccion
        type_combination (PredictionsCombinationType): Tipo de combinacion a aplicar sobre las predicciones

    Returns:
        list: Bounding boxes extraídas de la imagen, normalizados
    """
    # Create model
    model: SAE = getPredictionModel(dataset_name=dataset_name)

    if forwardParameters.val_dropout > 0:
        model.enable_eval_dropout()
        model.set_dropout_probability(dropout_probability=forwardParameters.val_dropout)
    
    image = image.resize(SAE_IMAGE_SIZE)
    
    transforms = get_transform()
    transformedImage = transforms(image)
    
    with no_grad():
        result = forwardToModel(model=model,
                                image=transformedImage,
                                times_pass_model=forwardParameters.times_pass_model,
                                type_combination=forwardParameters.type_combination
                                )

        boxes = getConnectedComponents(result, bin_threshold_percentaje=forwardParameters.bin_umbral)

        if forwardParameters.uses_redimension_vertical or forwardParameters.uses_redimension_horizontal:
            vResize = BBOX_REDIMENSIONED_RECOVER if forwardParameters.uses_redimension_vertical   else 1
            hResize = BBOX_REDIMENSIONED_RECOVER if forwardParameters.uses_redimension_horizontal else 1
            boxes = [resize_box(box, vResize=vResize, hResize=hResize) for box in boxes]
            
    x_max, y_max = SAE_IMAGE_SIZE
    boxes = [normalize_box(box, x_max, y_max) for box in boxes]

    return boxes