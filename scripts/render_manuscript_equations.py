"""Render the manuscript equations as high-resolution, transparent PNGs."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "publication" / "aoa" / "equations"


def save_formula(index: int, formula: str, width: float, height: float = 0.72) -> None:
    fig = plt.figure(figsize=(width, height), facecolor="white")
    fig.text(0.5, 0.5, f"${formula}$", ha="center", va="center", fontsize=18)
    fig.savefig(
        OUTPUT / f"equation_{index:02d}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


def save_decision_rule() -> None:
    fig = plt.figure(figsize=(8.8, 2.05), facecolor="white")
    fig.text(
        0.5,
        0.80,
        r"$\mathrm{NPV}=\dfrac{TN}{TN+FN}$",
        ha="center",
        va="center",
        fontsize=18,
    )
    fig.text(0.17, 0.39, r"$d(x)=$", ha="right", va="center", fontsize=18)
    fig.text(0.19, 0.39, "{", ha="center", va="center", fontsize=76, family="DejaVu Serif")
    rows = [
        (0.59, r"$\mathrm{approve},\quad \hat p(x)<\tau_A,$"),
        (0.39, r"$\mathrm{defer},\quad \tau_A\leq\hat p(x)<\tau_I,$"),
        (0.19, r"$\mathrm{investigate},\quad \hat p(x)\geq\tau_I.$"),
    ]
    for y, text in rows:
        fig.text(0.245, y, text, ha="left", va="center", fontsize=17)
    fig.savefig(
        OUTPUT / "equation_03.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_formula(
        1,
        r"F_2=\dfrac{5PR}{4P+R},\qquad P=\dfrac{TP}{TP+FP},\qquad R=\dfrac{TP}{TP+FN}.",
        8.0,
    )
    save_formula(
        2,
        r"p_k(x)=\dfrac{1}{1+\exp\!\left\{a_k\,z_k(x)+b_k\right\}},\qquad "
        r"\hat p(x)=\dfrac{1}{5}\sum_{k=1}^{5}p_k(x).",
        9.2,
    )
    save_decision_rule()
    save_formula(
        4,
        r"C=c_IN_I+c_LN_{I0}+c_F\!\left(N_{A1}+\rho N_{D1}\right)+c_DN_{D0},\qquad \rho=0.25.",
        9.2,
    )
    save_formula(
        5,
        r"I_j=\mathrm{PR\!\!-\!AUC}(f,X,y)-\mathrm{PR\!\!-\!AUC}(f,X_{\pi(j)},y).",
        8.5,
    )
    print(f"Rendered five equations to {OUTPUT}")


if __name__ == "__main__":
    main()
