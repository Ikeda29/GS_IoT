# GS_IoT
👥 Integrantes

Gustavo Ikeda (RM554718)

Henrique Azevedo (RM556707)

Isadora Meneghetti (RM556326)

Renato Cordão (RM556403)

Victoria Moura (RM555474)

📝 Descrição
Sistema de visão computacional para detecção automática de furacões em imagens e vídeos de satélite.

📚 Bibliotecas Utilizadas

OpenCV (cv2)	
NumPy	
scikit-learn
scikit-image
pickle

Instalação das dependências:
pip install opencv-python numpy scikit-learn scikit-image

Uso:
  python detector_furacao.py train - para treinar o modelo
  python detector_furacao.py detect videofuracao.mp4 - detectar furacão no video
  python detector_furacao.py run    videofuracao.mp4 - treinar e depois detectar

Pressione Q para encerrar o video antecipadamente
