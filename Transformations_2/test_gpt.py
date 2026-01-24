import cv2
import numpy as np
import os
import masque


# ============================================================
# TFCP robuste par PCA
# ============================================================

def TFCP_robuste(mask):
    ys, xs = np.where(mask == 255)

    if len(xs) < 100:
        raise ValueError("Masque vide ou trop petit")

    xf = xs.mean()
    yf = ys.mean()

    X = np.column_stack((xs - xf, ys - yf))

    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)

    return angle_deg, (xf, yf), principal_axis


# ============================================================
# Rotation image
# ============================================================

def realign_image(img, angle_deg):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================
# Dessin axe TFCP
# ============================================================

def draw_tfcp_axis(img, center, axis, color=(0, 0, 255), length=200):
    x0, y0 = int(center[0]), int(center[1])
    dx = axis[0] * length
    dy = axis[1] * length

    img_axis = img.copy()
    cv2.line(
        img_axis,
        (int(x0 - dx), int(y0 - dy)),
        (int(x0 + dx), int(y0 + dy)),
        color,
        2
    )
    cv2.circle(img_axis, (x0, y0), 4, (255, 0, 0), -1)

    return img_axis


# ============================================================
# MAIN
# ============================================================

def main(img_path, out_dir="Transformations_2/output"):

    os.makedirs(out_dir, exist_ok=True)

    mask_path = masque.main(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    angle, center, axis = TFCP_robuste(mask)
    print(f"[INFO] Angle TFCP brut = {angle:.2f} degrés")

    img = cv2.imread(img_path)

    # ----------------------------
    # AVANT : axe TFCP
    # ----------------------------
    img_before = draw_tfcp_axis(img, center, axis)

    # ----------------------------
    # ALIGNEMENT VERTICAL
    # ----------------------------
    aligned = realign_image(img, angle - 90)

    # axe vertical après
    axis_vertical = np.array([0.0, 1.0])
    center_aligned = (aligned.shape[1] // 2, aligned.shape[0] // 2)
    img_after = draw_tfcp_axis(aligned, center_aligned, axis_vertical)

    # ----------------------------
    # SAUVEGARDES
    # ----------------------------
    base = os.path.splitext(os.path.basename(img_path))[0]

    cv2.imwrite(os.path.join(out_dir, f"{base}_aligned_vertical.png"), aligned)

    canvas = np.hstack((img_before, img_after))
    cv2.imwrite(os.path.join(out_dir, f"{base}_before_after_tfcp_vertical.png"), canvas)

    print("[OK] Empreinte alignée verticalement")
    print("[OK] Images sauvegardées")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main("Transformations_2/input/empreinte_overlined_tournee.jpeg")
