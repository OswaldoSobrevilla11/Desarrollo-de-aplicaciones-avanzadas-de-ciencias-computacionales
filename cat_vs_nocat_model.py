#!/usr/bin/env python3
"""
cat_vs_nocat_model.py
Implementación desde cero (NumPy) del reto Cat vs No Cat.
Usa dataset/ para entrenar y predecir imágenes.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import resize
from sklearn.utils import shuffle

# =========================================================
# UTILIDADES DE ARCHIVOS Y PREPROCESAMIENTO
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_images_from_folder(folder, image_size=(64,64)):
    imgs, names = [], []
    if not os.path.exists(folder):
        return np.array(imgs), names
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(image_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            imgs.append(arr)
            names.append(fname)
        except Exception as e:
            print(f"Warning al abrir {path}: {e}")
    if len(imgs) == 0:
        return np.array([]), []
    return np.stack(imgs, axis=0), names

def preprocess_images(arr):
    if arr.size == 0:
        return np.zeros((0,0))
    m, h, w, c = arr.shape
    X = arr.reshape(m, -1).T / 255.0
    return X  # (n_px, m)

# =========================================================
# GENERADOR DE IMÁGENES SINTÉTICAS (para prueba)
# =========================================================
def generate_synthetic_cat_image(size=(64,64)):
    img = Image.new('RGB', size, (255,255,255))
    draw = ImageDraw.Draw(img)
    w,h = size
    cx, cy = w//2, h//2 + 4
    r = min(w,h)//3
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(200,160,120), outline=(0,0,0))
    ear_w = r//2
    ear_h = r//2
    draw.polygon([(cx-r+6, cy-r+10), (cx-r+6+ear_w, cy-r+10-ear_h), (cx-r+6+ear_w*2, cy-r+10)], fill=(200,160,120))
    draw.polygon([(cx+r-6, cy-r+10), (cx+r-6-ear_w, cy-r+10-ear_h), (cx+r-6-ear_w*2, cy-r+10)], fill=(200,160,120))
    draw.ellipse((cx- r//2 + 6, cy- r//6, cx- r//4 + 6, cy + r//6), fill=(0,0,0))
    draw.ellipse((cx+ r//4 - 6, cy- r//6, cx+ r//2 - 6, cy + r//6), fill=(0,0,0))
    draw.polygon([(cx, cy+6), (cx-6, cy+12), (cx+6, cy+12)], fill=(255,120,120))
    draw.line((cx, cy+12, cx, cy+20), fill=(0,0,0))
    draw.line((cx, cy+20, cx-8, cy+26), fill=(0,0,0))
    draw.line((cx, cy+20, cx+8, cy+26), fill=(0,0,0))
    return img

def generate_synthetic_nocat_image(size=(64,64)):
    img = Image.new('RGB', size, (255,255,255))
    draw = ImageDraw.Draw(img)
    w,h = size
    for i in range(0, h, 6):
        c = (np.random.randint(150,255), np.random.randint(150,255), np.random.randint(150,255))
        draw.rectangle([0,i,w,i+4], fill=c)
    for _ in range(6):
        x0 = np.random.randint(0,w-10)
        y0 = np.random.randint(0,h-10)
        x1 = x0 + np.random.randint(5, w//3)
        y1 = y0 + np.random.randint(5, h//3)
        c = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        draw.ellipse([x0,y0,x1,y1], fill=c, outline=None)
    return img

def generate_and_save_synthetic_images(cat_dir, nocat_dir, n_cat=20, n_nocat=20, size=(64,64)):
    ensure_dir(cat_dir)
    ensure_dir(nocat_dir)
    for i in range(n_cat):
        img = generate_synthetic_cat_image(size=size)
        img.save(os.path.join(cat_dir, f"synt_cat_{i:03d}.png"))
    for i in range(n_nocat):
        img = generate_synthetic_nocat_image(size=size)
        img.save(os.path.join(nocat_dir, f"synt_nocat_{i:03d}.png"))
    print(f"Generadas {n_cat} imágenes en {cat_dir} y {n_nocat} en {nocat_dir}")

# =========================================================
# RED NEURONAL (desde cero)
# =========================================================
def initialize_parameters(n_x, n_h, n_y, seed=1):
    np.random.seed(seed)
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def sigmoid(Z): return 1 / (1 + np.exp(-Z))
def relu(Z): return np.maximum(0, Z)
def relu_deriv(Z): return (Z > 0).astype(float)

def forward_propagation(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return A2, {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}

def compute_cost(A2, Y):
    m = Y.shape[1]
    eps = 1e-8
    cost = -(1/m) * np.sum(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))
    return np.squeeze(cost)

def backward_propagation(params, cache, X, Y):
    m = X.shape[1]
    W2 = params['W2']
    A1, A2 = cache['A1'], cache['A2']
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * relu_deriv(cache['Z1'])
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2}

def update_parameters(params, grads, lr=0.0075):
    for key in params.keys():
        if key in ['W1','W2']:
            params[key] -= lr * grads['d'+key]
        if key in ['b1','b2']:
            params[key] -= lr * grads['d'+key]
    return params

def predict(params, X):
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int), A2

# =========================================================
# ENTRENAMIENTO Y EVALUACIÓN
# =========================================================
def build_dataset_from_folders(train_cat, train_nocat, image_size=(64,64)):
    Xs, Ys = [], []
    Xc, _ = load_images_from_folder(train_cat, image_size=image_size)
    if Xc.size > 0:
        Xc = preprocess_images(Xc)
        Xs.append(Xc); Ys.append(np.ones((1, Xc.shape[1])))
    Xn, _ = load_images_from_folder(train_nocat, image_size=image_size)
    if Xn.size > 0:
        Xn = preprocess_images(Xn)
        Xs.append(Xn); Ys.append(np.zeros((1, Xn.shape[1])))
    if len(Xs)==0: return None, None
    X = np.concatenate(Xs, axis=1)
    Y = np.concatenate(Ys, axis=1)
    X, Y = shuffle(X.T, Y.T, random_state=1)
    return X.T, Y.T

def train(X, Y, n_h=20, iters=2000, lr=0.005):
    n_x, n_y = X.shape[0], 1
    params = initialize_parameters(n_x, n_h, n_y)
    for i in range(iters):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, lr)
        if i % 200 == 0:
            print(f"Iter {i}: coste = {cost:.6f}")
    return params

def evaluate_and_save_predictions(params, predict_folder, image_size=(64,64)):
    cat_dir = os.path.join(predict_folder, 'cat')
    nocat_dir = os.path.join(predict_folder, 'nocat')
    ensure_dir(cat_dir); ensure_dir(nocat_dir)
    imgs_cat, names_cat = load_images_from_folder(cat_dir, image_size=image_size)
    imgs_nocat, names_nocat = load_images_from_folder(nocat_dir, image_size=image_size)
    all_imgs, all_names, labels = [], [], []
    for arr, names, label in [(imgs_cat, names_cat, 1), (imgs_nocat, names_nocat, 0)]:
        for i in range(arr.shape[0]):
            all_imgs.append(arr[i]); all_names.append(names[i]); labels.append(label)
    if not all_imgs:
        print("No hay imágenes en dataset/predict/*")
        return
    A = np.stack(all_imgs, axis=0)
    X = preprocess_images(A)
    preds, probs = predict(params, X)
    acc = np.mean(preds.flatten() == np.array(labels))
    print(f"Exactitud sobre predict/: {acc*100:.2f}%")
    pred_cat_dir = os.path.join(predict_folder, 'pred_cat')
    pred_nocat_dir = os.path.join(predict_folder, 'pred_nocat')
    ensure_dir(pred_cat_dir); ensure_dir(pred_nocat_dir)
    for i, name in enumerate(all_names):
        pred = preds[0,i]
        prob = probs[0,i]
        img = Image.open(os.path.join(predict_folder, 'cat' if labels[i]==1 else 'nocat', name))
        save_dir = pred_cat_dir if pred==1 else pred_nocat_dir
        save_name = f"pred_{'cat' if pred==1 else 'nocat'}_{i:03d}_p{prob:.2f}_{name}"
        img.save(os.path.join(save_dir, save_name))
    print("Predicciones guardadas en pred_cat/ y pred_nocat/")

# =========================================================
# MAIN
# =========================================================
def main():
    train_cat = 'dataset/train/cat'
    train_nocat = 'dataset/train/nocat'
    predict_folder = 'dataset/predict'

    ensure_dir(train_cat)
    ensure_dir(train_nocat)

    # Generar dataset sintético si no hay imágenes
    if len(os.listdir(train_cat))==0 or len(os.listdir(train_nocat))==0:
        print("Generando dataset sintético de entrenamiento...")
        generate_and_save_synthetic_images(train_cat, train_nocat, n_cat=80, n_nocat=80)

    X_train, Y_train = build_dataset_from_folders(train_cat, train_nocat)
    print(f"Dataset de entrenamiento cargado: X={X_train.shape}, Y={Y_train.shape}")

    params = train(X_train, Y_train, n_h=20, iters=2000, lr=0.005)

    ensure_dir(predict_folder)
    ensure_dir(os.path.join(predict_folder, 'cat'))
    ensure_dir(os.path.join(predict_folder, 'nocat'))
    if len(os.listdir(os.path.join(predict_folder, 'cat')))==0 or len(os.listdir(os.path.join(predict_folder, 'nocat')))==0:
        print("Generando imágenes sintéticas en dataset/predict/...")
        generate_and_save_synthetic_images(os.path.join(predict_folder, 'cat'), os.path.join(predict_folder, 'nocat'))

    evaluate_and_save_predictions(params, predict_folder=predict_folder)

    preds, _ = predict(params, X_train)
    acc = float((preds == Y_train).mean())
    print(f"Exactitud en entrenamiento: {acc*100:.2f}%")

    ensure_dir('models')
    np.savez('models/cat_vs_nocat_params.npz', **params)
    print("Modelo guardado en models/cat_vs_nocat_params.npz")
    print("Entrenamiento y predicción completados correctamente.")

if __name__ == "__main__":
    main()
