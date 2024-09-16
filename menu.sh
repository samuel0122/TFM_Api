#!/bin/bash

# Función para mostrar el menú
mostrar_menu() {
    echo "Seleccione una opción:"
    echo "1) Actualizar dependencias"
    echo "2) Commit todos los cambios"
    echo "3) Probar API"
    echo "4) Enviar imagen a API"
    echo "5) Salir"
}

# Función para actualizar dependencias
actualizar_dependencias() {
    echo "Actualizando dependencias y generando requirements.txt..."
    pip freeze > requirements.txt
    echo "Dependencias actualizadas."
}

# Función para hacer commit de todos los cambios
commit_cambios() {
    read -p "Ingrese el mensaje para el commit: " mensaje
    echo "Haciendo commit de todos los cambios..."
    git add .
    git commit -m "$mensaje"
    echo "Enviando cambios al repositorio de Heroku..."
    
    # Ejecuta el comando y muestra todas las salidas en la terminal
    git push heroku master
    
    echo "Cambios comprometidos y enviados al repositorio de Heroku."
}

# Función para probar la API
probar_api() {
    echo "Probando API..."
    curl https://tfm-detector-pentagramas-api-a15aebd93b08.herokuapp.com/status
    echo ""
}

enviar_imagen_api() {
    echo "Enviando imagen CleanArchitecture.jpg a API con puerto 5001"
    curl -X POST http://localhost:5001/process-page \
      -F "pageId=123" \
      -F "database=456" \
      -F "page=@CleanArchitecture.jpg"
    echo""
}

# Iniciar .venv -> source .venv/bin/activate

# Bucle para mostrar el menú y capturar la opción seleccionada
while true; do
    mostrar_menu
    read -p "Ingrese una opción: " opcion

    case $opcion in
        1)
            actualizar_dependencias
            ;;
        2)
            commit_cambios
            ;;
        3)
            probar_api
            ;;
        4)
            enviar_imagen_api
            ;;
        5)
            echo "Saliendo..."
            break
            ;;
        *)
            echo "Opción no válida. Por favor, intente de nuevo."
            ;;
    esac

    echo ""
done
