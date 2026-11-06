import cv2
import numpy as np
import onnxruntime as ort

class PhotoRestorationPipeline:
    def __init__(self, onnx_model_path: str = None):
        if onnx_model_path:
            self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    def prevent_white_overflow(self, image: np.ndarray) -> np.ndarray:
        """Proteção contra estouro/clipagem de altas luzes (overflow de branco)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Aplica CLAHE com limite suave para evitar clipping
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_boosted = clahe.apply(l)
        # Compressão de tom suave nas zonas próximas de 255
        l_safe = np.where(l_boosted > 240, 240 + (l_boosted - 240) * 0.5, l_boosted).astype(np.uint8)
        merged = cv2.merge((l_safe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def create_hybrid_mask(self, image: np.ndarray) -> np.ndarray:
        """Gera máscara híbrida identificando rasgos, arranhões e variações de contraste."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detecção de estruturas finas / rasgos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        scratches = cv2.add(tophat, blackhat)
        
        _, mask_scratches = cv2.threshold(scratches, 15, 255, cv2.THRESH_BINARY)
        # Inpainting inicial leve na máscara
        restored_base = cv2.inpaint(image, mask_scratches, 3, cv2.INPAINT_TELEA)
        return restored_base

    def process(self, image_path: str, output_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Erro ao carregar a imagem")

        # 1. Aplica Máscara Híbrida e Inpainting básico
        step1 = self.create_hybrid_mask(img)

        # 2. Proteção de Overflows de Branco + Realce de Cor
        step2 = self.prevent_white_overflow(step1)

        # 3. Realce de Nitidez (Unsharp Masking)
        gaussian = cv2.GaussianBlur(step2, (0, 0), 2.0)
        sharpened = cv2.addWeighted(step2, 1.5, gaussian, -0.5, 0)

        # 4. Salva o resultado final
        cv2.imwrite(output_path, sharpened)
        return output_path