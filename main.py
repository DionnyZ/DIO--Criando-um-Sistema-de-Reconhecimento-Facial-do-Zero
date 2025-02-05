import cv2
import face_recognition
import numpy as np

# Carregar a imagem para ser reconhecida
image = cv2.imread("images/todos.jpg")

# Detectar rostos
face_locations = face_recognition.face_locations(image, model="hog")

# Codificar rostos detectados
face_encodings = face_recognition.face_encodings(image, face_locations)

# Carregar imagens dos rostos conhecidos e obter as codificações faciais
codificacao_pessoa1 = face_recognition.face_encodings(face_recognition.load_image_file("images/tyler_james_williams_1.jpg"))[0]
codificacao_pessoa2 = face_recognition.face_encodings(face_recognition.load_image_file("images/tequan_richmond_1.jpg"))[0]
codificacao_pessoa3 = face_recognition.face_encodings(face_recognition.load_image_file("images/imani_hakim_1.jpg"))[0]
codificacao_pessoa4 = face_recognition.face_encodings(face_recognition.load_image_file("images/terry_crews_1.jpg"))[0]
codificacao_pessoa5 = face_recognition.face_encodings(face_recognition.load_image_file("images/tichina_arnold_1.jpg"))[0]
codificacao_pessoa6 = face_recognition.face_encodings(face_recognition.load_image_file("images/vincent_martella_1.png"))[0]

# Armazenar as codificações
known_face_encodings = [codificacao_pessoa1, codificacao_pessoa2, codificacao_pessoa3, codificacao_pessoa4, codificacao_pessoa5, codificacao_pessoa6]
known_face_names = ["Tyler James", "Tequan Richmond", "Imani Hakim", "Terry Crews", "Tichina Arnold", "Vincent Martella"]

# Identificação dos rostos
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Desconhecido"

    # Identificar os rostos
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Adicionar um retângulo ao redor do rosto
    cv2.rectangle(image, (left, top), (right, bottom), (30, 255, 30), 2)
    
    # Adicionar o nome da pessoa
    cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 255, 30), 2)

# Mostrar resultado
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()