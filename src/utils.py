import os

from src.defines import API_IMAGES_FOLDERS

def get_dataset_folder(dataset):
    """Devuelve la carpeta correspondiente al dataset"""
    folder = API_IMAGES_FOLDERS.get(dataset.lower())
    if not folder:
        return None
    return folder

def get_mimetype(filename):
    """Devuelve el MIME type en función de la extensión del archivo"""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.gif':
        return 'image/gif'
    else:
        return None