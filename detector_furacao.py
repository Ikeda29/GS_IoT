"""
Detector de Furacao com Imagem de Satélite

Gustavo Ikeda (RM554718)
Henrique Azevedo (RM556707)
Isadora Meneghetti (RM556326)
Renato Cordão (RM556403)
Victoria Moura (RM555474)

Uso:
  python detector_furacao.py train
  python detector_furacao.py detect videofuracao.mp4
  python detector_furacao.py run    videofuracao.mp4

Pressione Q para encerrar.
"""

import cv2
import numpy as np
import pickle
import sys
from pathlib import Path

from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

# ── Configuracoes ─────────────────────────────────────────────────
PATCH_SIZE  = 64          
DATASET_DIR = "dados"
MODEL_FILE  = "modelo_furacao.pkl"
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}

# ── Deteccao ──────────────────────────────────────────────────────
class Deteccao:
    def __init__(self, confianca, centro, raio, frame, tempo):
        self.confianca = confianca
        self.centro    = centro
        self.raio      = raio
        self.frame     = frame
        self.tempo     = tempo

# ── Extrator de Features ──────────────────────────────────────────
class Extrator:

    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            (PATCH_SIZE, PATCH_SIZE), (16,16), (8,8), (8,8), 9
        )

    def _prep(self, img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, (PATCH_SIZE, PATCH_SIZE))
        return cv2.createCLAHE(2.0, (4,4)).apply(g)

    def _hog(self, g):
        f = self.hog.compute(g)
        return f.ravel() if f is not None else np.zeros(144)

    def _lbp(self, g):
        lbp = local_binary_pattern(g, 8, 1, "uniform")
        h, _ = np.histogram(lbp.ravel(), 10, (0,10), density=True)
        return h.astype(np.float32)

    def _fisicas(self, g):
        h, w = g.shape
        cx, cy, r = w//2, h//2, min(w,h)//2 - 1
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

        # Espiral: gradiente tangencial
        sp, sw = [], []
        for ri in np.linspace(r*0.2, r*0.85, 5):
            for t in np.linspace(0, 2*np.pi, 24):
                px, py = int(cx+ri*np.cos(t)), int(cy+ri*np.sin(t))
                if 0<=px<w and 0<=py<h:
                    mag = float(np.hypot(gx[py,px], gy[py,px]))
                    if mag > 2:
                        sp.append(abs(float(gx[py,px])*(-np.sin(t)) +
                                       float(gy[py,px])*np.cos(t)) / mag)
                        sw.append(mag)
        espiral = float(np.average(sp, weights=sw)) if sp else 0.0

        # Gradiente radial
        aneis = []
        for ri in np.linspace(r*0.1, r*0.85, 6):
            vals = [int(g[int(cy+ri*np.sin(t)), int(cx+ri*np.cos(t))])
                    for t in np.linspace(0, 2*np.pi, 20)
                    if 0<=int(cx+ri*np.cos(t))<w and 0<=int(cy+ri*np.sin(t))<h]
            if vals: aneis.append(float(np.mean(vals)))
        if len(aneis) >= 3:
            slope = float(np.polyfit(np.arange(len(aneis)), aneis, 1)[0])
            grad_r = float(np.clip(-slope/(np.ptp(aneis)+1e-6), -1, 1))
        else:
            grad_r = 0.0

        # Cobertura nubosa e brilho medio
        mask = np.zeros_like(g); cv2.circle(mask,(cx,cy),r,255,-1)
        v = g[mask==255]
        blob   = float(np.mean(v > np.mean(g)*1.05))
        brilho = float(np.mean(v))/255.0

        return np.array([espiral, grad_r, blob, brilho], dtype=np.float32)

    def extrair(self, img: np.ndarray) -> np.ndarray:
        g = self._prep(img)
        return np.concatenate([self._hog(g), self._lbp(g),
                                self._fisicas(g)]).astype(np.float32)


# ── Modelo ────────────────────────────────────────────────────────
class Modelo:
    PESOS = [0.45, 0.35, 0.20]

    def __init__(self):
        self.extrator  = Extrator()
        self.pipelines = []
        self.threshold = 0.45
        self.treinado  = False

    def _pipes(self):
        return [
            Pipeline([("sc", StandardScaler()),
                      ("clf", GradientBoostingClassifier(
                          n_estimators=100, max_depth=4,
                          learning_rate=0.1, random_state=42))]),
            Pipeline([("sc", StandardScaler()),
                      ("clf", RandomForestClassifier(
                          n_estimators=100, n_jobs=-1, random_state=42))]),
            Pipeline([("sc", StandardScaler()),
                      ("clf", CalibratedClassifierCV(
                          SVC(kernel="rbf", C=8, gamma="scale"), cv=3))]),
        ]

    def treinar(self, X, y):
        print(f"\n  Treinando: {len(X)} amostras "
              f"({int(y.sum())} furacao / {int((y==0).sum())} nao-furacao)")
        self.pipelines = self._pipes()
        for nome, pipe in zip(["GradientBoosting","RandomForest","SVM"],
                               self.pipelines):
            auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
            pipe.fit(X, y)
            print(f"  {nome}: AUC = {auc:.3f}")

        # Otimiza threshold pelo F1
        probs = self._proba(X)
        melhor_f1, melhor_thr = 0.0, 0.45
        for thr in np.arange(0.25, 0.80, 0.02):
            pred = (probs >= thr).astype(int)
            tp=np.sum((pred==1)&(y==1)); fp=np.sum((pred==1)&(y==0))
            fn=np.sum((pred==0)&(y==1))
            prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
            f1=2*prec*rec/(prec+rec+1e-8)
            if f1 > melhor_f1: melhor_f1, melhor_thr = f1, thr

        self.threshold = melhor_thr
        self.treinado  = True
        print(f"  Threshold: {self.threshold:.2f}  (F1={melhor_f1:.3f})")

    def _proba(self, X):
        probs = np.zeros(len(X))
        for pipe, w in zip(self.pipelines, self.PESOS):
            try:    probs += w * pipe.predict_proba(X)[:, 1]
            except: probs += w * pipe.predict(X).astype(float)
        return probs

    def prever(self, X): return self._proba(X)

    def salvar(self, path):
        with open(path,"wb") as f:
            pickle.dump({"pipes":self.pipelines,"thr":self.threshold},f)
        print(f"  Modelo salvo: {path}")

    def carregar(self, path):
        with open(path,"rb") as f: d=pickle.load(f)
        self.pipelines=d["pipes"]; self.threshold=d["thr"]; self.treinado=True
        print(f"  Modelo carregado: {path}  (threshold={self.threshold:.2f})")


# ── Dataset ───────────────────────────────────────────────────────
def carregar_dados(pasta, extrator):
    base=Path(pasta); pos=base/"furacao"; neg=base/"nao_furacao"
    if not pos.exists():
        print(f"[ERRO] Pasta nao encontrada: {pos}")
        print(f"  Crie: {pasta}/furacao/  e  {pasta}/nao_furacao/")
        sys.exit(1)

    X, y = [], []
    def ler(folder, label):
        imgs=[p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
        print(f"  [{folder.name}] {len(imgs)} imagens")
        for p in imgs:
            img=cv2.imread(str(p))
            if img is None: continue
            variants=[img, cv2.flip(img,1), cv2.flip(img,0),
                      cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE),
                      np.clip(img.astype(np.float32)*1.2,0,255).astype(np.uint8),
                      np.clip(img.astype(np.float32)*0.8,0,255).astype(np.uint8)]
            for v in variants:
                try: X.append(extrator.extrair(v)); y.append(label)
                except: pass

    ler(pos, 1)
    if neg.exists():
        ler(neg, 0)
    else:
        print("  [AVISO] nao_furacao/ nao encontrada - gerando negativos sinteticos")
        rng=np.random.default_rng(0)
        for _ in range(len(X)):
            img=rng.integers(20,100,(PATCH_SIZE,PATCH_SIZE,3),dtype=np.uint8)
            try: X.append(extrator.extrair(img)); y.append(0)
            except: pass

    idx=np.random.default_rng(42).permutation(len(y))
    return np.array(X,dtype=np.float32)[idx], np.array(y)[idx]


# ── Detector em Video ─────────────────────────────────────────────
class Detector:
    def __init__(self, modelo):
        self.modelo    = modelo
        self.historico = {}

    # ── Pre-filtro rapido ─────────────────────────────────────────
    def _candidatos(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape

        # Blur para suavizar ruido
        blur = cv2.GaussianBlur(gray, (15,15), 0)

        # Limiar adaptativo: pixels mais brilhantes que a media
        thr_val = max(100, int(np.mean(blur) * 1.1))
        _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)

        # Morfologia: fecha buracos, remove ruido pequeno
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

        candidatos = []
        min_area = (h * w) * 0.02   # ignora blobs < 2% do frame

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Raio = media entre equivalente circular e circulo inscrito
            r_equiv = int(np.sqrt(area / np.pi))
            _, enc_r = cv2.minEnclosingCircle(cnt)
            r = int(np.clip((r_equiv + enc_r) / 8,
                             min(h,w)//40, min(h,w)//10))
            candidatos.append((cx, cy, r))

        # Fallback: se nao encontrou nada, testa o centro do frame
        if not candidatos:
            r = min(h,w)//16
            candidatos.append((w//2, h//2, r))

        return candidatos, gray

    def _suavizar(self, cx, cy, conf):
        k=(cx//80, cy//80)
        hist=self.historico.setdefault(k,[])
        hist.append(conf); hist[:]=hist[-5:]
        w=np.arange(1,len(hist)+1,dtype=float)**1.5
        return float(np.average(hist,weights=w))

    def _nms(self, dets):
        dets=sorted(dets,key=lambda d:d.confianca,reverse=True)
        kept=[]
        for d in dets:
            if not any(np.hypot(d.centro[0]-k.centro[0],
                                d.centro[1]-k.centro[1])<max(d.raio,k.raio)*0.6
                       for k in kept):
                kept.append(d)
        return kept

    def analisar(self, frame, frame_num, fps):
        h, w = frame.shape[:2]
        candidatos, _ = self._candidatos(frame)

        # Extrai feature apenas dos candidatos (poucos — tipicamente 1 a 5)
        validos, feats = [], []
        for cx, cy, r in candidatos:
            y1,y2=max(0,cy-r),min(h,cy+r)
            x1,x2=max(0,cx-r),min(w,cx+r)
            patch=frame[y1:y2,x1:x2]
            if patch.shape[0]<16 or patch.shape[1]<16: continue
            try:
                feats.append(self.modelo.extrator.extrair(patch))
                validos.append((cx,cy,r))
            except: pass

        if not feats: return []

        probs=self.modelo.prever(np.array(feats,dtype=np.float32))
        dets=[]
        for (cx,cy,r), prob in zip(validos, probs):
            conf=self._suavizar(cx,cy,float(prob))
            if conf < self.modelo.threshold: continue
            dets.append(Deteccao(confianca=conf,
                                  frame=frame_num,tempo=frame_num/max(fps,1),
                                  centro=(cx,cy),raio=r))
        return self._nms(dets)

    def desenhar(self, frame, dets, frame_num):
        out=frame.copy(); h,w=out.shape[:2]
        bar=out.copy()
        cv2.rectangle(bar,(0,0),(w,52),(8,10,14),-1)
        cv2.addWeighted(bar,0.72,out,0.28,0,out)
        cv2.putText(out,f"DETECTOR FURACAO  |  Frame {frame_num:05d}  |  [Q] Sair",
            (10,18),cv2.FONT_HERSHEY_SIMPLEX,0.50,(200,230,255),1,cv2.LINE_AA)
        status=f"{len(dets)} FURACAO(ES) DETECTADO(S)" if dets else "Monitorando..."
        cv2.putText(out,status,(10,38),cv2.FONT_HERSHEY_SIMPLEX,0.45,
            (30,70,255) if dets else (90,190,90),1,cv2.LINE_AA)

        for d in dets:
            cor=(0, int(255*(1-d.confianca)), 255) if d.confianca < 0.7 else (0,0,255)
            cx,cy,r=d.centro[0],d.centro[1],d.raio
            cv2.circle(out,(cx,cy),r//2,cor,2)
            cv2.circle(out,(cx,cy),max(3,r//20),cor,1)
            cv2.drawMarker(out,(cx,cy),cor,cv2.MARKER_CROSS,16,2)
            label=f"FURACAO  {d.confianca:.0%}"
            lx,ly=max(0,cx-r),max(cy-r-8,55)
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            cv2.rectangle(out,(lx,ly-th-3),(lx+tw+4,ly+2),(10,12,16),-1)
            cv2.putText(out,label,(lx+2,ly-1),cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,cor,1,cv2.LINE_AA)


        return out


# ── Funcoes principais ────────────────────────────────────────────
def treinar():
    print("\n=== TREINAMENTO ===")
    ext=Extrator()
    X,y=carregar_dados(DATASET_DIR,ext)
    mod=Modelo(); mod.extrator=ext
    mod.treinar(X,y)
    mod.salvar(MODEL_FILE)
    print("Treinamento concluido!\n")
    return mod


def detectar(video_path, modelo=None):
    print("\n=== DETECCAO ===")
    if modelo is None:
        modelo=Modelo()
        if not Path(MODEL_FILE).exists():
            print(f"[ERRO] Modelo nao encontrado: {MODEL_FILE}")
            print("  Execute primeiro: python detector_furacao.py train")
            sys.exit(1)
        modelo.carregar(MODEL_FILE)

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRO] Nao foi possivel abrir: {video_path}"); sys.exit(1)

    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {video_path}  |  {w}x{h} @ {fps:.0f}fps")
    print("  Pressione Q para encerrar.\n")

    det=Detector(modelo)
    frame_n=0; total_dets=0
    ultimo_resultado=[]
    ultimo_analise=-9999

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame_n+=1

        # Analisa apenas 1 vez por segundo
        if frame_n - ultimo_analise >= fps:
            ultimo_analise = frame_n
            ultimo_resultado = det.analisar(frame, frame_n, fps)
            total_dets += len(ultimo_resultado)
            for d in ultimo_resultado:
                print(f"  \033[91m[FURACAO]\033[0m  conf={d.confianca:.0%}  frame={frame_n}  t={d.tempo:.1f}s")

        anotado=det.desenhar(frame, ultimo_resultado, frame_n)
        dw=min(1280,max(w,400)); dh=min(720,max(h,300))
        cv2.imshow("Detector Furacao [Q=Sair]",
                   cv2.resize(anotado,(dw,dh)))

        key=cv2.waitKey(1)&0xFF
        if key in (ord("q"),ord("Q"),27):
            print("  Encerrado pelo usuario."); break
    cap.release(); cv2.destroyAllWindows()
    print(f"\n  Total de deteccoes: {total_dets}\n")


# ── Entry point ───────────────────────────────────────────────────
def main():
    if len(sys.argv)<2:
        print(__doc__); sys.exit(0)
    cmd=sys.argv[1].lower()
    if   cmd=="train":  treinar()
    elif cmd=="detect":
        if len(sys.argv)<3:
            print("Uso: python detector_furacao.py detect <video.mp4>"); sys.exit(1)
        detectar(sys.argv[2])
    elif cmd=="run":
        if len(sys.argv)<3:
            print("Uso: python detector_furacao.py run <video.mp4>"); sys.exit(1)
        detectar(sys.argv[2], treinar())
    else:
        print(f"Comando desconhecido: {cmd}")
        print("Comandos: train | detect <video> | run <video>")

if __name__=="__main__":
    main()
