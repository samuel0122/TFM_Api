def resize_box(box, vResize, hResize):
    """
      Boxes as [x1, y1, x2, y2]
    """
    if(len(box)) < 4:
      print(box)
    # Get dimensions
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1 # 100
    h = y2 - y1 # 100

    # Calculate resized dimensions
    w2 = w * hResize # 80
    h2 = h * vResize # 80

    # Get size difference
    wDif = w2 - w  # 100 - 80 = 20
    hDif = h2 - h  # 100 - 80 = 20

    # Get how much each coordinate must move
    xMov = wDif / 2
    yMov = hDif / 2

    # Get new coordinates
    x1 = x1 - xMov
    x2 = x2 + xMov

    y1 = y1 - yMov
    y2 = y2 + yMov

    return [x1, y1, x2, y2]

def normalize_box(box, x_max, y_max):
    """
    Normaliza las coordenadas de una caja [x1, y1, x2, y2] 
    a valores entre 0 y 1 basados en los valores máximos x_max y y_max.
    
    Args:
        box (list): Lista con las coordenadas [x1, y1, x2, y2].
        x_max (float): Valor máximo en la dimensión x.
        y_max (float): Valor máximo en la dimensión y.
        
    Returns:
        list: Caja normalizada con coordenadas [x1', y1', x2', y2'] entre 0 y 1.
    """
    if len(box) < 4:
        print("Box has fewer than 4 elements:", box)
        return box
    
    # Extraer coordenadas
    x1, y1, x2, y2 = box
    
    # Normalizar las coordenadas
    x1_norm = x1 / x_max
    y1_norm = y1 / y_max
    x2_norm = x2 / x_max
    y2_norm = y2 / y_max
    
    return [x1_norm, y1_norm, x2_norm, y2_norm]
