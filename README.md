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
Biblioteca	Versão	Utilização
OpenCV (cv2)	≥ 4.5	Processamento de imagem, vídeo, HOG
NumPy	≥ 1.19	Operações matriciais e cálculos
scikit-learn	≥ 0.24	Classificadores ML (RandomForest, SVM, GradientBoosting)
scikit-image	≥ 0.18	Feature LBP (Local Binary Pattern)
pickle	-	Persistência do modelo treinado

Instalação das dependências:
pip install opencv-python numpy scikit-learn scikit-image

Uso:
  python detector_furacao.py train - para treinar o modelo
  python detector_furacao.py detect videofuracao.mp4 - detectar furacão no video
  python detector_furacao.py run    videofuracao.mp4 - treinar e depois detectar

Pressione Q para encerrar o video antecipadamente
