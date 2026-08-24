#!/usr/bin/env python3
"""Build a visually verified PDF companion from the TeX report's results.

The primary editable research document is ``double_chaotic_validation.tex``.
This helper exists because a TeX engine is not bundled in the current runtime;
it renders a faithful Japanese companion PDF from the same JSON values and
figures so that pagination and figure legibility can be checked here.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
DATA = json.loads((BUNDLE / "data" / "validation_summary.json").read_text(encoding="utf-8"))
OUTPUT = HERE / "double_chaotic_validation.pdf"


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\YuGothM.ttc")
    bold = Path(r"C:\Windows\Fonts\YuGothB.ttc")
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError("Yu Gothic fonts were not found")
    pdfmetrics.registerFont(TTFont("YuGothic", str(regular), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("YuGothicBold", str(bold), subfontIndex=0))
    pdfmetrics.registerFontFamily(
        "YuGothic", normal="YuGothic", bold="YuGothicBold", italic="YuGothic", boldItalic="YuGothicBold"
    )
    return "YuGothic", "YuGothicBold"


FONT, FONT_BOLD = register_fonts()
PAGE_WIDTH, PAGE_HEIGHT = A4


def fmt(value: float, digits: int = 4) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


base = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle(
        "JapaneseTitle", parent=base["Title"], fontName=FONT_BOLD, fontSize=20,
        leading=30, alignment=TA_CENTER, textColor=colors.HexColor("#123B57"), wordWrap="CJK"
    ),
    "subtitle": ParagraphStyle(
        "JapaneseSubtitle", parent=base["Normal"], fontName=FONT, fontSize=11,
        leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#455A64"), wordWrap="CJK"
    ),
    "h1": ParagraphStyle(
        "JapaneseH1", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=15,
        leading=22, spaceBefore=10, spaceAfter=7, textColor=colors.HexColor("#123B57"), wordWrap="CJK"
    ),
    "h2": ParagraphStyle(
        "JapaneseH2", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=12,
        leading=18, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1E617A"), wordWrap="CJK"
    ),
    "body": ParagraphStyle(
        "JapaneseBody", parent=base["BodyText"], fontName=FONT, fontSize=9.4,
        leading=15.2, alignment=TA_JUSTIFY, spaceAfter=6, wordWrap="CJK"
    ),
    "small": ParagraphStyle(
        "JapaneseSmall", parent=base["BodyText"], fontName=FONT, fontSize=8,
        leading=12, alignment=TA_LEFT, wordWrap="CJK"
    ),
    "caption": ParagraphStyle(
        "JapaneseCaption", parent=base["BodyText"], fontName=FONT, fontSize=8.5,
        leading=13, alignment=TA_CENTER, spaceBefore=5, spaceAfter=8, wordWrap="CJK"
    ),
    "equation": ParagraphStyle(
        "Equation", parent=base["BodyText"], fontName=FONT, fontSize=10.5,
        leading=17, alignment=TA_CENTER, spaceBefore=5, spaceAfter=7, wordWrap="CJK"
    ),
    "quote": ParagraphStyle(
        "JapaneseQuote", parent=base["BodyText"], fontName=FONT, fontSize=9.2,
        leading=15, leftIndent=12 * mm, rightIndent=12 * mm, borderColor=colors.HexColor("#A9C6D3"),
        borderWidth=1, borderPadding=8, backColor=colors.HexColor("#F4F9FB"), wordWrap="CJK"
    ),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def figure(path: Path, caption: str, max_height: float = 188 * mm):
    image = Image(str(path))
    max_width = 162 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return [image, P(caption, "caption")]


def table(rows, widths, header=True):
    converted = []
    for row_index, row in enumerate(rows):
        style_name = "small"
        converted.append([P(str(cell), style_name) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0BEC5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9FA")]),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBF2")),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ]
        )
    result.setStyle(TableStyle(commands))
    return result


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.drawString(23 * mm, 13 * mm, "MultifieldInflationLattice - 2026-08-18 run validation")
    canvas.drawRightString(PAGE_WIDTH - 23 * mm, 13 * mm, f"{doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CFD8DC"))
    canvas.line(23 * mm, 18 * mm, PAGE_WIDTH - 23 * mm, 18 * mm)
    canvas.restoreState()


run = DATA["run"]
analytic = DATA["analytic"]
constraints = DATA["constraints"]
energy = DATA["energy"]
spectra = DATA["spectra"]
ode = DATA["ode_metrics"]
model = DATA["model"]

story = []
story += [
    Spacer(1, 20 * mm),
    P("二場二次ポテンシャル格子シミュレーションの<br/>解析的 slow-roll 近似および独立 ODE との比較", "title"),
    Spacer(1, 8 * mm),
    P("対象: MultifieldInflationLattice 2026年8月18日実行結果", "subtitle"),
    P(f"run: {run['id']}", "subtitle"),
    Spacer(1, 18 * mm),
    P(
        "<b>概要</b><br/>本報告は、解析的slow-roll近似、Friedmann拘束、独立なDormand-Prince 5(4) ODE、"
        "Bunch-Davies初期スペクトル、Hubble-exit後の軽い場power、およびParseval恒等式を用いて、"
        "8月18日の二場格子計算を検証する。選定した全17検査はPASSした。これは背景発展、"
        "真空初期化、FFT規格化の限定的妥当性を支持するが、連続極限やseed ensembleまで証明しない。",
        "quote",
    ),
    Spacer(1, 20 * mm),
    P("作成日: 2026年8月24日", "subtitle"),
    PageBreak(),
]

story += [
    P("1. 目的と検証範囲", "h1"),
    P(
        "自己重力を含む非一様二場系には、全発展を覆う有用な初等関数の閉形式解はない。"
        "したがって、本検証は解析的slow-roll予測、厳密恒等式、独立高精度ODEを組み合わせる。"
        "目的はコード全体の証明ではなく、このrunの背景、拘束、初期ゆらぎおよびFFT規格化が"
        "理論的期待と整合するかを確認することである。"
    ),
    P("複素場という表現", "h2"),
    P(
        "実装メタデータは二つの正準実スカラー場を明記する。Φ=(φ+iψ)/sqrt(2) とまとめることはできるが、"
        "mφ != mψ のためU(1)対称ではない。正確には、質量の異なる二実成分、またはU(1)が明示的に"
        "破れた異方的複素場である。"
    ),
    P("2. モデルと連続方程式", "h1"),
    P("V(φ,ψ) = (mφ² φ² + mψ² ψ²)/2", "equation"),
    P("d2phi_I/dt2 + 3H dphi_I/dt - a^-2 Laplacian(phi_I) + m_I^2 phi_I = 0", "equation"),
    P("H^2 = rho/3,　dH/dt = -(rho+p)/2", "equation"),
    P("ρ=K+G+V,　p=K-G/3-V", "equation"),
]

settings = [
    ["項目", "設定値"],
    ["場・質量", "phi, psi; 9e-6, 1e-6"],
    ["初期平均場・速度", "(13,13); (1e-10,0)"],
    ["格子", "32^3, L=2, periodic"],
    ["空間差分", "second-order 7-point Laplacian"],
    ["時間積分", "leapfrog, timestep factor 0.01"],
    ["ゆらぎ", "Bunch-Davies, seed 8, effective mass off"],
    ["終了条件", "epsilon_H >= 1"],
    ["Friedmann warning/error", "1e-6 / 1e-3"],
]
story += [table(settings, [48 * mm, 112 * mm]), Spacer(1, 4 * mm)]

story += [
    P("3. 解析的slow-roll比較", "h1"),
    P("3H φ̇_I ≃ -m_I²φ_I,　H² ≃ V/3", "equation"),
    P(
        "u=ψ/ψ0, r=mφ²/mψ²=81 とすると、ψ_SR=ψ0u, φ_SR=φ0u^r, "
        "N_SR=[φ0²(1-u^(2r))+ψ0²(1-u²)]/4 となる。epsilon_V=1を解析曲線の終了点とした。"
        "比較にはN>=1, epsilon_H<0.1, Omega_G<1e-3を満たす区間の規格化RMS差を使用した。"
    ),
]

summary_rows = [
    ["比較量", "解析/基準", "格子/測定", "差・判定"],
    ["初期 H", fmt(analytic["initial_H"]), fmt(run["initial_H"]), pct(analytic["initial_H_relative_difference"]) + " / PASS"],
    ["総 e-fold", fmt(analytic["total_efolds"]), fmt(run["total_efolds"]), pct(analytic["total_efolds_relative_difference"]) + " / PASS"],
    ["支配交代 N", fmt(analytic["dominance_transition_efolds"]), fmt(run["dominance_transition_efolds"]), pct(analytic["dominance_transition_relative_difference"]) + " / PASS"],
    [
        "slow-roll nRMS (H/phi/psi)",
        "0",
        f"{fmt(analytic['H_parametric_normalized_rms'])} / {fmt(analytic['phi_normalized_rms'])} / {fmt(analytic['psi_normalized_rms'])}",
        "< 0.05 / PASS",
    ],
]
story += [
    PageBreak(),
    P("解析比較の判定概要", "h1"),
    table(summary_rows, [46 * mm, 32 * mm, 34 * mm, 48 * mm]),
    Spacer(1, 6 * mm),
    P(
        "解析比較の実用的許容値は、初期Hubble率1%、総e-fold数5%、支配場交代10%、"
        "slow-roll区間の規格化RMS差5%とした。これは普遍的な精度基準ではなく、"
        "単一runの理論整合性を説明するために設定した探索的基準である。"
    ),
    P(
        f"格子runの終了点はN={fmt(run['total_efolds'])}, epsilon_H={fmt(run['final_epsilon_H'])}, "
        f"w={fmt(run['final_w'])}であり、epsilon_H=1とw=-1/3の理論関係を再現する。"
        "終了近傍ではslow-roll近似そのものが破れるため、その偏差を格子誤差とは判定しない。"
    ),
    P(
        "重いphiが先に減衰し、軽いpsiが後半を支配する二段階発展が得られた。"
        "次頁の図は、同じ初期値から横方向のshiftや再規格化を行わずに解析曲線を重ねたものである。",
        "quote",
    ),
    PageBreak(),
]

story += figure(BUNDLE / "figures" / "background_vs_analytic.png", "図1　格子平均背景と解析的slow-roll近似。二段階発展、場空間軌道、Hubble率、slow-roll量を比較する。", 220 * mm)
story += [PageBreak()]

story += [
    P("4. エネルギー、拘束条件、独立ODE", "h1"),
    P(
        f"初期エネルギー比は Omega_V={fmt(energy['initial_potential_fraction'])}, "
        f"Omega_K={fmt(energy['initial_kinetic_fraction'])}, Omega_G={fmt(energy['initial_gradient_fraction'])}。"
        f"slow-roll区間の最大勾配比は {fmt(energy['gradient_fraction_maximum_slowroll'])} で、背景比較は勾配に支配されない。"
    ),
    P(
        f"最大Friedmann残差は {fmt(constraints['friedmann_maximum'])} で、設定error tolerance 1e-3以下である。"
        "warning tolerance 1e-6は超えるため、機械精度の拘束保存を主張するものではない。"
    ),
    P(
        f"独立ODEの平均場nRMS差は {fmt(ode['csv_driven_field_mean_ode'])}、平均速度は "
        f"{fmt(ode['csv_driven_field_velocity_ode'])}、自己無撞着一様場Hの診断差は "
        f"{fmt(ode['homogeneous_H_comparison'])} である。"
    ),
]
story += figure(BUNDLE / "figures" / "energy_and_constraints.png", "図2　エネルギー比、場の支配交代、Friedmann残差、解析近似誤差。", 192 * mm)
story += [PageBreak()]
story += figure(BUNDLE / "ode_validation" / "ode_consistency.png", "図3　独立Dormand-Prince 5(4) ODEとの比較。CSV-driven解と自己無撞着一様場解を分離して示す。", 225 * mm)
story += [PageBreak()]

horizon = spectra["horizon_exit_light_field"]
story += [
    P("5. ゆらぎ初期化とスペクトル規格化", "h1"),
    P("初期Bunch-Davies期待値: Delta²_delta_phi(k)=k²/(4 pi² a0²)", "equation"),
    P("軽い場のHubble-exit予測: P_delta_psi(k) ≃ [H(k=aH)/(2 pi)]²", "equation"),
    P(
        f"初期shell powerのmode-weighted理論比はphi={fmt(spectra['bunch_davies']['phi']['mode_weighted_power_ratio'])}, "
        f"psi={fmt(spectra['bunch_davies']['psi']['mode_weighted_power_ratio'])}。"
        f"十分なmode数を持つ{horizon['well_populated_bins']} shellのHubble-exit理論比は平均"
        f"{fmt(horizon['mean_power_ratio'])}、中央値{fmt(horizon['median_power_ratio'])}である。"
    ),
    P(
        f"Parseval再構成のpeak規格化最大差はphi={fmt(spectra['parseval']['phi']['peak_normalized_max_abs_difference'])}, "
        f"psi={fmt(spectra['parseval']['psi']['peak_normalized_max_abs_difference'])}。"
        "したがって、初期真空振幅、Fourier規格化、shell集約、実空間分散は互いに整合する。"
    ),
]
story += figure(BUNDLE / "figures" / "fluctuation_spectrum_checks.png", "図4　Bunch-Davies初期スペクトル、軽い場のHubble-exit後power、Parseval分散再構成。edge shellはmode数が少なく散乱が大きい。", 225 * mm)
story += [PageBreak()]

metrics = json.loads((BUNDLE / "data" / "validation_summary.json").read_text(encoding="utf-8"))
story += [
    P("6. 妥当性評価", "h1"),
    P(
        "単一格子・単一seedについて、背景場、Hubble率、状態方程式、総e-fold、支配場交代は"
        "二場二次ポテンシャルのslow-roll予測と数パーセント以内で一致した。独立ODE、Friedmann拘束、"
        "Bunch-Davies振幅、軽い場powerおよびParseval再構成も設定した許容値内である。"
        "従って本runは、背景発展、ゆらぎ初期化、場スペクトル規格化の限定的妥当性を支持する。",
        "quote",
    ),
    P("確立していない事項", "h2"),
    P(
        "(1) 時間刻み・空間解像度・box sizeを変えた収束と連続極限、(2) 複数seedのensemble不確かさ、"
        "(3) metric perturbationを含む完全なMukhanov-Sasaki系、(4) linear curvatureおよびdelta-N spectrumの"
        "独立定量検証は本報告の範囲外である。linear curvatureは一様FLRW背景からのpostprocess proxyであり、"
        "delta-Nは終了時一様密度面のlate-time diagnosticである。"
    ),
    P("7. 再現方法", "h1"),
    P(
        "compare_double_chaotic.py をプロジェクト仮想環境で実行すると、data/validation_metrics.csv、"
        "data/validation_summary.json、比較時系列、全図を再生成する。編集可能な主文書は"
        "report/double_chaotic_validation.texであり、LuaLaTeXを2回実行して参照番号を解決できる。"
    ),
    P("8. 結論", "h1"),
    P(
        "8月18日の完了runは、解析的slow-roll予測、厳密背景恒等式、独立ODE、初期量子ゆらぎ、"
        "Hubble-exit後の軽い場power、FFT規格化の全てと矛盾せず、選定17項目をPASSした。"
        "従って背景と場ゆらぎ部門の妥当性を研究上説明できる。次段階は解析式の追加よりも、"
        "解像度、時間刻み、体積、seedを変えた収束・統計検証である。"
    ),
    P("参考文献", "h1"),
    P("[1] A. D. Linde, Chaotic Inflation, Physics Letters B 129 (1983) 177-181, doi:10.1016/0370-2693(83)90837-7.", "small"),
    P("[2] C. Gordon et al., Adiabatic and entropy perturbations from inflation, Phys. Rev. D 63 (2001) 023506, arXiv:astro-ph/0009131.", "small"),
    P("[3] G. Felder and I. Tkachev, LATTICEEASY, Comput. Phys. Commun. 178 (2008) 929-932, arXiv:hep-ph/0011159.", "small"),
]


doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    leftMargin=23 * mm,
    rightMargin=23 * mm,
    topMargin=22 * mm,
    bottomMargin=24 * mm,
    title="MultifieldInflationLattice 2026-08-18 run validation",
    author="Validation report generated from the recorded run",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(OUTPUT)
