import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import masque


# ============================================================
# TFCP robuste par PCA  (code original inchangé)
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
# Rotation image  (code original inchangé)
# ============================================================

def realign_image(img, angle_deg):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# Utilitaires visualisation
# ============================================================

def to_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_axis_on_ax(ax, cx, cy, axis, length=180, color='red',
                    label=None, linestyle='-'):
    dx = axis[0] * length
    dy = axis[1] * length
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy],
            color=color, lw=2, linestyle=linestyle,
            label=label, zorder=5)
    ax.plot(cx, cy, 'o', color='blue', markersize=6, zorder=6)


def draw_angle_annotation(ax, cx, cy, angle_deg, radius=90):
    a_start = np.radians(90)
    a_end   = np.radians(angle_deg)
    theta   = np.linspace(a_start, a_end, 100)
    ax.plot(cx + radius * np.cos(theta),
            cy - radius * np.sin(theta),
            color='orange', lw=2, zorder=5)
    mid = (a_start + a_end) / 2
    deviation = abs(angle_deg - 90)
    ax.text(cx + (radius + 14) * np.cos(mid),
            cy - (radius + 14) * np.sin(mid),
            f'{deviation:.1f}°', color='orange',
            fontsize=9, fontweight='bold', ha='center', zorder=7)


def panel_title(ax, text, color='#2563EB'):
    ax.set_title(text, fontsize=9, fontweight='bold', color='white', pad=5,
                 bbox=dict(boxstyle='round,pad=0.35',
                           facecolor=color, edgecolor='none'))


# ============================================================
# FIGURE 1 — Avant / Après  (2 panneaux)
# ============================================================

def fig_avant_apres(img, aligned, angle_deg, axis, center, out_dir, base):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#F1F5F9')
    fig.suptitle("Réalignement de l'empreinte digitale — Avant / Après",
                 fontsize=13, fontweight='bold', y=1.02)

    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    # ── Avant ────────────────────────────────────────────────
    axes[0].imshow(to_rgb(img))
    draw_axis_on_ax(axes[0], cx, cy, axis, length=200,
                    color='red', label='Axe PCA')
    axes[0].plot([cx, cx], [cy - 200, cy + 200],
                 'b--', lw=1.5, alpha=0.8, label='Référence verticale')
    draw_angle_annotation(axes[0], cx, cy, angle_deg, radius=100)
    axes[0].legend(fontsize=8, loc='lower right',
                   framealpha=0.85, edgecolor='#CBD5E1')
    panel_title(axes[0], 'Avant correction', '#64748B')
    axes[0].set_xlim(0, w); axes[0].set_ylim(h, 0)
    axes[0].axis('off')

    # ── Après ─────────────────────────────────────────────────
    cxa, cya = aligned.shape[1] // 2, aligned.shape[0] // 2
    axes[1].imshow(to_rgb(aligned))
    axes[1].plot([cxa, cxa], [cya - 200, cya + 200],
                 'b-', lw=2, alpha=0.9, label='Axe vertical (corrigé)')
    axes[1].plot(cxa, cya, 'o', color='blue', markersize=6)
    axes[1].legend(fontsize=8, loc='lower right',
                   framealpha=0.85, edgecolor='#CBD5E1')
    panel_title(axes[1],
                f'Après correction  (rotation {angle_deg - 90:.2f}°)', '#2563EB')
    axes[1].set_xlim(0, aligned.shape[1])
    axes[1].set_ylim(aligned.shape[0], 0)
    axes[1].axis('off')

    fig.text(0.505, 0.50, '→', fontsize=30, color='#2563EB',
             ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{base}_fig1_avant_apres.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#F1F5F9')
    plt.show()
    print(f'  {path}')


# ============================================================
# FIGURE 2 — Pipeline complet 6 étapes
# ============================================================

def fig_pipeline(img, mask, aligned, angle_deg, axis, center, out_dir, base):
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    img_mid   = realign_image(img, (angle_deg - 90) / 2)
    diff_gray = cv2.cvtColor(cv2.absdiff(img, aligned), cv2.COLOR_BGR2GRAY)

    overlay            = to_rgb(img).copy()
    mask_color         = np.zeros_like(overlay)
    mask_color[:, :, 0] = mask
    overlay = cv2.addWeighted(overlay, 0.6, mask_color, 0.4, 0)

    fig = plt.figure(figsize=(18, 7), facecolor='#F1F5F9')
    fig.suptitle('Pipeline de réalignement — vue détaillée',
                 fontsize=13, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.06,
                           left=0.02, right=0.98,
                           top=0.88, bottom=0.04)

    panels = [
        (gs[0, 0], to_rgb(img),     '1. Image originale',                        None,   '#64748B'),
        (gs[0, 1], mask,            '2. Masque (segmentation PCA)',               'gray', '#0F766E'),
        (gs[0, 2], overlay,         '3. Superposition masque / image',            None,   '#7C3AED'),
        (gs[1, 0], to_rgb(img_mid), f'4. Rotation ×½  ({(angle_deg-90)/2:.1f}°)', None,  '#92400E'),
        (gs[1, 1], to_rgb(aligned), f'5. Image alignée  ({angle_deg-90:.2f}°)',   None,  '#2563EB'),
        (gs[1, 2], diff_gray,       '6. Différence absolue',                     'hot',  '#DC2626'),
    ]

    for spec, data, title, cmap, color in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(data, cmap=cmap)
        panel_title(ax, title, color)
        ax.axis('off')

        if title.startswith('1.'):
            draw_axis_on_ax(ax, cx, cy, axis, length=180, color='red')
            ax.plot([cx, cx], [cy - 180, cy + 180], 'b--', lw=1.2, alpha=0.7)
        if title.startswith('5.'):
            cxa, cya = aligned.shape[1] // 2, aligned.shape[0] // 2
            ax.plot([cxa, cxa], [cya - 180, cya + 180], 'b-', lw=1.8, alpha=0.85)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{base}_fig2_pipeline.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#F1F5F9')
    plt.show()
    print(f'  ✔  {path}')


# ============================================================
# FIGURE 3 — Géométrie annotée de la rotation
# ============================================================

def fig_geometrie(img, aligned, angle_deg, axis, center, out_dir, base):
    h, w = img.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    cxa, cya = aligned.shape[1] // 2, aligned.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#F1F5F9')
    fig.suptitle('Transformation géométrique — détail',
                 fontsize=13, fontweight='bold', y=1.02)

    # Superposition transparente
    blend = cv2.addWeighted(to_rgb(img), 0.5, to_rgb(aligned), 0.5, 0)
    axes[0].imshow(blend)
    panel_title(axes[0], 'Superposition avant/après', '#7C3AED')
    axes[0].axis('off')

    # Original annoté
    axes[1].imshow(to_rgb(img))
    draw_axis_on_ax(axes[1], cx, cy, axis, length=200,
                    color='red', label='Axe PCA (TFCP)')
    axes[1].plot([cx, cx], [cy - 200, cy + 200],
                 'b--', lw=1.5, alpha=0.85, label='Référence verticale')
    draw_angle_annotation(axes[1], cx, cy, angle_deg, radius=110)
    axes[1].legend(fontsize=8, loc='lower right', framealpha=0.85)
    panel_title(axes[1], f'Axe PCA détecté : {angle_deg:.2f}°', '#DC2626')
    axes[1].set_xlim(0, w); axes[1].set_ylim(h, 0)
    axes[1].axis('off')

    # Corrigé annoté
    axes[2].imshow(to_rgb(aligned))
    axes[2].plot([cxa, cxa], [cya - 200, cya + 200],
                 'b-', lw=2, alpha=0.9, label='Axe vertical corrigé')
    axes[2].plot(cxa, cya, 'o', color='blue', markersize=6)
    axes[2].text(cxa + 8, cya - 210, 'Axe\ncorrigé',
                 color='blue', fontsize=8, fontweight='bold')
    axes[2].legend(fontsize=8, loc='lower right', framealpha=0.85)
    panel_title(axes[2],
                f'Après rotation de {angle_deg - 90:.2f}°', '#2563EB')
    axes[2].set_xlim(0, aligned.shape[1])
    axes[2].set_ylim(aligned.shape[0], 0)
    axes[2].axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{base}_fig3_geometrie.png')
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='#F1F5F9')
    plt.show()
    print(f'  ✔  {path}')


# ============================================================
# MAIN
# ============================================================

def main(img_path, out_dir='Transformations_2/output'):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]

    mask_path = masque.main(img_path)
    mask      = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img       = cv2.imread(img_path)

    angle, center, axis = TFCP_robuste(mask)
    print(f'[INFO] Angle TFCP brut     = {angle:.2f}°')
    print(f'[INFO] Rotation appliquée  = {angle - 90:.2f}°')

    aligned = realign_image(img, angle - 90)

    # Sauvegarde originale conservée
    cv2.imwrite(os.path.join(out_dir, f'{base}_aligned_vertical.png'), aligned)

    # Figures rapport
    print('\n[INFO] Génération des figures rapport…')
    fig_avant_apres(img, aligned, angle, axis, center, out_dir, base)
    fig_pipeline   (img, mask, aligned, angle, axis, center, out_dir, base)
    fig_geometrie  (img, aligned, angle, axis, center, out_dir, base)

    print(f'\n 3 figures sauvegardées dans « {out_dir}/ »')


# ============================================================
# EXECUTION
# ============================================================

if __name__ == '__main__':
    main('Transformations_2/input/Syria1.jpeg')