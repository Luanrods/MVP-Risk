# build_report.py
from __future__ import annotations
 
import argparse
import os
import sys
 
import numpy as np
import pandas as pd
 
from src.simulation import simulate_cost_risk
from src.metrics import summary_metrics, risk_driver_table, add_emv
from src.charts import plot_histogram, plot_s_curve, plot_risk_drivers
 
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_report.py",
        description=(
            "Gera um relatório PDF de Análise Quantitativa de Riscos (QCRA) "
            "a partir de um registro de riscos em CSV, usando simulação de Monte Carlo."
        ),
    )
    parser.add_argument(
        "--input", "-i", default="data/example_risk_register.csv",
        help="Caminho do CSV com o registro de riscos (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-cost", type=float, default=40_000_000.0,
        help="Custo baseline do projeto, sem riscos (default: %(default)s)",
    )
    parser.add_argument(
        "--budget", type=float, default=44_000_000.0,
        help="Orçamento / limite de aprovação (default: %(default)s)",
    )
    parser.add_argument(
        "--n-simulations", type=int, default=100_000,
        help="Número de iterações da simulação Monte Carlo (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reprodutibilidade dos resultados (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="outputs",
        help="Pasta onde salvar o PDF e os gráficos gerados (default: %(default)s)",
    )
    return parser.parse_args()
 
 
def fail(message: str) -> None:
    """Imprime um erro amigável no stderr e encerra com código de saída 1."""
    print(f"Erro: {message}", file=sys.stderr)
    sys.exit(1)
 
 
def main() -> None:
    args = parse_args()
 
    # -----------------------------------------------------------------
    # 1. Carrega e valida o registro de riscos
    # -----------------------------------------------------------------
    if not os.path.exists(args.input):
        fail(f"arquivo de entrada não encontrado: {args.input}")
 
    try:
        risks = pd.read_csv(args.input)
    except Exception as exc:
        fail(f"não foi possível ler o CSV de entrada ({args.input}): {exc}")
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    # -----------------------------------------------------------------
    # 2. Motor Monte Carlo + métricas
    # -----------------------------------------------------------------
    try:
        final_cost, contributions = simulate_cost_risk(
            df=risks, baseline_cost=args.baseline_cost,
            n_simulations=args.n_simulations, seed=args.seed,
        )
    except ValueError as exc:
        print("Erro no registro de riscos:", file=sys.stderr)
        for issue in str(exc).split(" | "):
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(1)
 
    metrics = summary_metrics(final_cost, baseline_cost=args.baseline_cost, budget=args.budget)
    drivers = risk_driver_table(contributions, final_cost)
    risks_emv = add_emv(risks)
 
    # -----------------------------------------------------------------
    # 3. Gráficos (salvos como PNG dentro de output_dir/)
    # -----------------------------------------------------------------
    fig1 = plot_histogram(final_cost, metrics, args.baseline_cost)
    fig1.savefig(os.path.join(args.output_dir, "hist.png"), dpi=170)
 
    fig2 = plot_s_curve(final_cost, budget=args.budget)
    fig2.savefig(os.path.join(args.output_dir, "scurve.png"), dpi=170)
 
    fig3 = plot_risk_drivers(drivers)
    fig3.savefig(os.path.join(args.output_dir, "drivers.png"), dpi=170)
 
    # -----------------------------------------------------------------
    # 4. Montagem do PDF
    # -----------------------------------------------------------------
    out_path = os.path.join(args.output_dir, "MVP-Risk.pdf")
 
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
 
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20,
                                  textColor=colors.HexColor("#12414a"), spaceAfter=4)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10.5,
                                     textColor=colors.HexColor("#555555"), spaceAfter=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13.5,
                         textColor=colors.HexColor("#12414a"), spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("BodyText2", parent=styles["Normal"], fontSize=9.7, leading=13.5)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8.3, leading=11.5,
                            textColor=colors.HexColor("#555555"))
    caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.3,
                              textColor=colors.HexColor("#666666"), alignment=1, spaceAfter=10)
 
    story = []
 
    story.append(Paragraph("MVP-Risk — Relatório de Caso Simulado", title_style))
    story.append(Paragraph(
        f"Análise Quantitativa de Riscos · Monte Carlo (Bernoulli + PERT/Triangular) · "
        f"{args.n_simulations:,} cenários · seed={args.seed}",
        subtitle_style,
    ))
 
    story.append(Paragraph("1. Premissas do caso", h2))
    premise_data = [
        ["Baseline cost", f"R$ {args.baseline_cost:,.0f}"],
        ["Budget / limite de aprovação", f"R$ {args.budget:,.0f}"],
        ["Número de simulações", f"{args.n_simulations:,}"],
        ["Seed", str(args.seed)],
        ["Riscos modelados", f"{len(risks)} ({(risks['type'] == 'risco').sum()} riscos, "
                              f"{(risks['type'] == 'oportunidade').sum()} oportunidades)"],
    ]
    t = Table(premise_data, colWidths=[7.5 * cm, 8.5 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#12414a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
    ]))
    story.append(t)
 
    story.append(Paragraph("2. Registro de Riscos", h2))
    reg_header = ["ID", "Tipo", "Descrição", "Prob.", "Dist.", "Min", "ML", "Max"]
    reg_rows = [reg_header]
    for _, r in risks.iterrows():
        reg_rows.append([
            r["id"], r["type"], r["description"], f"{r['probability']:.0%}",
            r["distribution"],
            f"{r['min_impact']:,.0f}", f"{r['most_likely_impact']:,.0f}", f"{r['max_impact']:,.0f}",
        ])
    reg_table = Table(reg_rows, colWidths=[1.3*cm, 2.1*cm, 3.9*cm, 1.3*cm, 1.7*cm, 2.0*cm, 2.0*cm, 2.0*cm])
    reg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12414a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f6")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(reg_table)
    story.append(Paragraph(
        "Riscos somam custo quando ocorrem; Oportunidades subtraem. Todas as magnitudes de "
        "impacto são inseridas como valores positivos — o sinal é controlado pela coluna type.",
        small,
    ))
 
    story.append(Paragraph("3. Resultados-chave", h2))
    kpi_data = [
        ["Mean", "P50", "P80", "P90", "P95"],
        [f"R$ {metrics['mean']/1e6:,.2f} mi", f"R$ {metrics['p50']/1e6:,.2f} mi",
         f"R$ {metrics['p80']/1e6:,.2f} mi", f"R$ {metrics['p90']/1e6:,.2f} mi",
         f"R$ {metrics['p95']/1e6:,.2f} mi"],
    ]
    kpi_table = Table(kpi_data, colWidths=[3.1*cm]*5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12414a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f2f6f6")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))
 
    extra_data = [
        ["Contingência P80 (vs. baseline)", f"R$ {metrics['p80_contingency']:,.0f}  "
         f"({metrics['p80_contingency']/args.baseline_cost:.1%} do baseline)"],
        ["Probabilidade de ficar dentro do budget", f"{metrics['prob_within_budget']:.1%}"],
        ["Desvio padrão", f"R$ {metrics['std_dev']:,.0f}"],
    ]
    t2 = Table(extra_data, colWidths=[8.5*cm, 7.5*cm])
    t2.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#12414a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
    ]))
    story.append(t2)
 
    story.append(PageBreak())
 
    story.append(Paragraph("4. Distribuição do custo final", h2))
    story.append(Image(os.path.join(args.output_dir, "hist.png"), width=16.5*cm, height=8.4*cm))
    story.append(Paragraph("Figura 1 — Histograma do custo final com P50/P80/P90 marcados.", caption))
 
    story.append(Paragraph("5. Curva S / CDF", h2))
    story.append(Image(os.path.join(args.output_dir, "scurve.png"), width=16.5*cm, height=8.4*cm))
    story.append(Paragraph(
        "Figura 2 — Probabilidade acumulada de o custo final ficar abaixo de cada valor no eixo X. "
        "A linha tracejada marca o budget informado.", caption,
    ))
 
    story.append(Paragraph("6. Risk drivers (sensibilidade)", h2))
    story.append(Image(os.path.join(args.output_dir, "drivers.png"), width=16.5*cm, height=8.4*cm))
    story.append(Paragraph(
        "Figura 3 — Correlação de Spearman entre a contribuição de cada risco e o custo final. "
        "Mede associação/sensibilidade, não causalidade nem magnitude monetária isolada.", caption,
    ))
 
    story.append(Paragraph("7. Expected Monetary Value (referência determinística)", h2))
    emv_rows = [["ID", "Tipo", "Impacto médio", "EMV (com sinal)"]]
    for _, r in risks_emv.iterrows():
        emv_rows.append([
            r["id"], r["type"], f"R$ {r['mean_impact']:,.0f}", f"R$ {r['signed_emv']:,.0f}",
        ])
    emv_rows.append(["", "", "Total EMV:", f"R$ {risks_emv['signed_emv'].sum():,.0f}"])
    emv_table = Table(emv_rows, colWidths=[2.2*cm, 3.2*cm, 5.5*cm, 5.6*cm])
    emv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12414a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f2f6f6")]),
        ("LINEABOVE", (0, -1), (-1, -1), 0.6, colors.HexColor("#12414a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -2), 0.3, colors.HexColor("#dddddd")),
    ]))
    story.append(emv_table)
    story.append(Paragraph(
        "O EMV é uma referência determinística simples (probabilidade × impacto médio, com sinal). "
        "Ele não substitui a distribuição probabilística — serve apenas como ponto de comparação.", small,
    ))
 
    story.append(Spacer(1, 14))
    story.append(Paragraph("8. Premissas e limitações do modelo", h2))
    limitations = [
        "Os eventos de risco são modelados como independentes entre si (sem correlação) na v0.1.",
        "A probabilidade representa a chance de ocorrência dentro do horizonte analisado.",
        "A distribuição descreve a magnitude do impacto condicional à ocorrência do evento.",
        "O modelo não inclui incerteza do baseline por padrão — apenas os eventos do risk register.",
        "Os resultados dependem diretamente da qualidade das estimativas de probabilidade e impacto.",
        "A seed fixa garante reprodutibilidade do cálculo, não previsibilidade do futuro.",
    ]
    for item in limitations:
        story.append(Paragraph(f"• {item}", body))
 
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Gerado com MVP-Risk v0.1 — MVP educacional e de portfólio para Quantitative Cost Risk Analysis.",
        small,
    ))
 
    doc.build(story)
    print(f"PDF gerado em: {os.path.abspath(out_path)}")
 
 
if __name__ == "__main__":
    main()