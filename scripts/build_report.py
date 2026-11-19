"""Build a three-page proposal and a complete PDF report from frozen evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                               PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import engineer_features, fixed_split, load_data, write_json
from scripts.build_bibliography import apa

OUT = ROOT / "output/pdf"
FIG = ROOT / "artifacts/report_figures"
WIDTH = A4[0] - 40 * mm
CLASSES = ["Dropout", "Enrolled", "Graduate"]
PALETTE = ["#B45309", "#2563EB", "#087F8C"]
TEAL = colors.HexColor("#087F8C")
TEAM = [member["name_en"] for member in json.loads((ROOT / "project_config.json").read_text())["team"]]


def normalise(text: str) -> str:
    return text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")


def strip_links(text: str) -> str:
    """Keep Markdown link labels; accept parentheses inside a DOI URL."""
    while True:
        match = re.search(r"\[([^\]]+)\]\(", text)
        if not match:
            return text
        end, depth = match.end(), 1
        while end < len(text) and depth:
            if text[end] == "(": depth += 1
            if text[end] == ")": depth -= 1
            end += 1
        if depth:
            return text
        text = text[:match.start()] + match.group(1) + text[end:]


def markup(text: str) -> str:
    text = html.escape(normalise(strip_links(text)))
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)


def register_styles() -> dict:
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in (("Study", "DejaVuSans.ttf"), ("StudyBold", "DejaVuSans-Bold.ttf"),
                           ("StudyItalic", "DejaVuSans-Oblique.ttf"), ("StudyBoldItalic", "DejaVuSans-BoldOblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Study", normal="Study", bold="StudyBold", italic="StudyItalic", boldItalic="StudyBoldItalic")
    base = dict(fontName="Study", textColor=colors.HexColor("#1F2937"), splitLongWords=True)
    return {
        "body": ParagraphStyle("Body", fontSize=9.9, leading=14, spaceAfter=7, **base),
        "proposal": ParagraphStyle("Proposal", fontSize=9.5, leading=13.1, spaceAfter=6, **base),
        "h1": ParagraphStyle("H1", fontSize=15, leading=19, spaceBefore=13, spaceAfter=9, keepWithNext=True, **{**base, "fontName": "StudyBold"}),
        "h2": ParagraphStyle("H2", fontSize=11.4, leading=15, spaceBefore=9, spaceAfter=6, keepWithNext=True, **{**base, "fontName": "StudyBold"}),
        "title": ParagraphStyle("Title", fontSize=24, leading=31, spaceAfter=17, **{**base, "fontName": "StudyBold"}),
        "subtitle": ParagraphStyle("Subtitle", fontSize=12, leading=17, spaceAfter=12, **base),
        "caption": ParagraphStyle("Caption", fontSize=8.2, leading=11, spaceAfter=11, **base),
        "table": ParagraphStyle("Cell", fontSize=8.1, leading=10.7, **base),
        "header": ParagraphStyle("CellHeader", fontSize=8.2, leading=10.8, textColor=colors.white, fontName="StudyBold"),
        "bibliography": ParagraphStyle("Bibliography", fontSize=8.3, leading=11.2, spaceAfter=8,
                                        leftIndent=12, firstLineIndent=-12, **base),
        "proposal_ref": ParagraphStyle("ProposalReference", fontSize=7.4, leading=9.5, spaceAfter=5,
                                       leftIndent=9, firstLineIndent=-9, **base),
    }


STYLES = register_styles()


def p(text: str, style="body") -> Paragraph:
    return Paragraph(markup(text), STYLES[style])


def heading(text: str, key: str | None = None, level=1) -> Paragraph:
    item = p(text, "h1" if level == 1 else "h2")
    item.section_key = key
    return item


class StudyPDF(BaseDocTemplate):
    def __init__(self, path: Path, label: str):
        super().__init__(str(path), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                         topMargin=20*mm, bottomMargin=18*mm,
                         title=label, author="; ".join(TEAM))
        self.section_pages = {}
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal", leftPadding=0, rightPadding=0)
        self.addPageTemplates(PageTemplate(id="study", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Study", 7.4)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(20*mm, A4[1]-12*mm, "Student Outcome Classification and Academic Profile Analysis")
        canvas.setStrokeColor(colors.HexColor("#D8E2E8"))
        canvas.line(20*mm, A4[1]-14*mm, A4[0]-20*mm, A4[1]-14*mm)
        canvas.drawString(20*mm, 10*mm, "AI-assisted draft for team review | 19 November 2026")
        canvas.drawRightString(A4[0]-20*mm, 10*mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        key = getattr(flowable, "section_key", None)
        if key and key not in self.section_pages:
            self.section_pages[key] = self.page


def paragraphs(text: str, style="body") -> list:
    return [p(" ".join(part.splitlines()), style) for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def sections(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i+1].strip() for i in range(1, len(parts), 2)}


def table(headers: list[str], rows: list[list], widths=None) -> Table:
    data = [[p(str(value), "header") for value in headers]]
    data += [[p(str(value), "table") for value in row] for row in rows]
    widths = [WIDTH / len(headers)] * len(headers) if widths is None else [WIDTH * item / sum(widths) for item in widths]
    # Every report table is short enough for one page; keep its rows together.
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=0)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
        ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
    ]))
    return result


def figure(name: str, caption: str, maximum_height=295) -> KeepTogether:
    path = FIG / f"{name}.png"
    with PILImage.open(path) as picture:
        width, height = picture.size
    scale = min(WIDTH / width, maximum_height / height)
    return KeepTogether([Spacer(1, 7), Image(str(path), width=width*scale, height=height*scale),
                         Spacer(1, 5), p(caption, "caption")])


def make_figures(summary: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = engineer_features(load_data())
    split, _ = fixed_split(data)
    train = data.loc[split["train"]]
    root = ROOT / "artifacts/tables"

    def save(fig, name):
        fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False, "axes.spines.right":False}):
        counts = train.Target.value_counts().reindex(CLASSES)
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        bars = ax.bar(CLASSES, counts, color=PALETTE)
        ax.bar_label(bars, labels=[f"{n:,} ({n/len(train):.1%})" for n in counts], padding=4)
        ax.set(ylim=(0, counts.max()*1.2), ylabel="Students", title="Class distribution - training only")
        save(fig,"eda_classes")

        fig,ax=plt.subplots(figsize=(6.5,3.5))
        boxes=ax.boxplot([train.loc[train.Target==c,"Age at enrollment"] for c in CLASSES],tick_labels=CLASSES,patch_artist=True)
        for box,color in zip(boxes['boxes'],PALETTE):box.set_facecolor(color);box.set_alpha(.55)
        ax.set(ylabel="Age at enrollment",title="Age and outcome - training only")
        save(fig,"eda_age")

        fig,ax=plt.subplots(figsize=(6.5,3.5))
        for c,color in zip(CLASSES,PALETTE):
            ax.hist(train.loc[train.Target==c,"Curricular units 1st sem (approved)"], bins=np.arange(-.5,27.5),
                    histtype="step",density=True,linewidth=1.8,label=c,color=color)
        ax.set(xlabel="Passed units, semester 1",ylabel="Within-class density",title="Academic progress - training only")
        ax.legend(fontsize=9);save(fig,"eda_passed")

        corr=pd.read_csv(root/'training_correlations.csv',index_col=0)
        labels=['Age','Admission','Enrolled','Evaluations','Passed','Mean grade','Pass rate']
        fig,ax=plt.subplots(figsize=(6.5,5.3))
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(7),labels,rotation=40,ha='right');ax.set_yticks(range(7),labels)
        for i in range(7):
            for j in range(7):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8.5,color='white' if abs(corr.iloc[i,j])>.65 else '#111827')
        fig.colorbar(im,ax=ax,shrink=.8,label='Pearson correlation')
        ax.set_title('Quantitative relationships - training only');save(fig,'eda_correlations')

        courses=pd.read_csv(root/'training_course_outcomes.csv',index_col=0)
        courses=courses.loc[courses.sum(axis=1).sort_values(ascending=False).index]
        proportions=courses.div(courses.sum(axis=1),axis=0)
        fig,ax=plt.subplots(figsize=(6.5,6.1));left=np.zeros(len(courses))
        for c,color in zip(CLASSES,PALETTE):
            ax.barh(range(len(courses)),proportions[c],left=left,color=color,label=c);left+=proportions[c].to_numpy()
        ax.set_yticks(range(len(courses)),[f'{c} (n={int(courses.loc[c].sum())})' for c in courses.index],fontsize=8.7)
        ax.invert_yaxis();ax.set(xlabel='Proportion within programme',ylabel='Programme code',xlim=(0,1),title='Programme outcomes - training only')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3,fontsize=8.5);save(fig,'eda_courses')

        comparison=pd.read_csv(root/'final_comparison.csv');x=np.arange(len(comparison))
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        ax.bar(x-.18,comparison.validation_macro_f1,width=.36,color='#087F8C',label='Validation')
        ax.bar(x+.18,comparison.test_macro_f1,width=.36,color='#2563EB',label='Test')
        ax.set_xticks(x,['Dummy','Logistic\nRegression','Decision\nTree','KNN','ANN'])
        ax.set(ylim=(0,1),ylabel='Macro F1',title='Main experiment - first-semester profile');ax.legend(fontsize=9)
        save(fig,'main_comparison')

        hist=pd.read_csv(root/'ann_history.csv');hist=hist[hist.candidate==summary['ann']['selected_candidate']]
        fig,axes=plt.subplots(2,1,figsize=(6.5,5.8),sharex=True)
        for ax,train_col,val_col,label in [(axes[0],'train_log_loss','validation_log_loss','Cross-entropy'),
                                          (axes[1],'train_macro_f1','validation_macro_f1','Macro F1')]:
            ax.plot(hist.epoch,hist[train_col],color='#087F8C',label='Train')
            ax.plot(hist.epoch,hist[val_col],color='#B45309',label='Validation')
            ax.axvline(summary['ann']['best_epoch'],color='#64748B',linestyle='--',label='Restored epoch')
            ax.set_ylabel(label);ax.legend(fontsize=8.3)
        axes[0].set_title('Selected ANN - observed training history');axes[1].set_xlabel('Epoch')
        fig.tight_layout();save(fig,'ann_history')

        ablation=pd.read_csv(root/'feature_ablation.csv');x=np.arange(3)
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        ax.bar(x-.18,ablation.validation_macro_f1,width=.36,color='#087F8C',label='Validation')
        ax.bar(x+.18,ablation.test_macro_f1,width=.36,color='#2563EB',label='Test')
        ax.set_xticks(x,['No semester\nresults','+ Semester 1','+ Semester 2'])
        ax.set(ylim=(0,1),ylabel='Macro F1',title='Logistic Regression - feature-group comparison');ax.legend(fontsize=9)
        save(fig,'ablation')

        variance=pd.read_csv(root/'pca_variance.csv')
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        ax.bar(variance.component,variance.explained_variance_ratio,color='#8FA8C8',label='Individual')
        ax.plot(variance.component,variance.cumulative_variance,'o-',color='#087F8C',label='Cumulative')
        ax.axhline(.95,color='#B45309',linestyle='--',label='95%');ax.set_xticks(variance.component)
        ax.set(ylim=(0,1.05),xlabel='Component',ylabel='Explained variance ratio',title='PCA - five academic indicators');ax.legend(fontsize=9)
        save(fig,'pca_variance')

        projected=pd.read_csv(root/'training_pca_projection.csv')
        for by,name in [('cluster','pca_clusters'),('Target','pca_classes')]:
            fig,ax=plt.subplots(figsize=(6.5,3.8))
            values=sorted(projected.cluster.unique()) if by=='cluster' else CLASSES
            for value,color in zip(values,['#475569','#B45309','#087F8C'] if by=='cluster' else PALETTE):
                part=projected[projected[by]==value]
                ax.scatter(part.PC1,part.PC2,s=8,alpha=.4,color=color,label=str(value),rasterized=True)
            ax.set(xlabel=f'PC1 ({variance.explained_variance_ratio.iloc[0]:.1%})',ylabel=f'PC2 ({variance.explained_variance_ratio.iloc[1]:.1%})',title='Training projection by '+('cluster' if by=='cluster' else 'outcome'))
            ax.legend(fontsize=8.5);save(fig,name)


def proposal_content(sources: dict, standalone: bool) -> list:
    text=(ROOT/'docs/research_proposal.md').read_text()
    pages=text.split('<!-- pagebreak -->')
    story=[]
    for i,part in enumerate(pages):
        if i:story.append(PageBreak())
        if i==0:
            story.extend([p('Research proposal','subtitle'),p('Stability of Student Outcome Classification','h1'),
                          p(', '.join(TEAM),'caption')])
        chunks=re.split(r'^## (.+)$',part,flags=re.MULTILINE)
        for j in range(1,len(chunks),2):
            title=chunks[j].strip()
            if title=='Proposal references':continue
            story.append(heading(title, 'proposal' if i==0 else None, level=2))
            story.extend(paragraphs(chunks[j+1],style='proposal'))
        if i==2:
            story.append(heading('Proposal references',level=2))
            for key in ['cawley2010','glandorf2024','realinho2022','setiawan2026','vaarma2024']:
                story.append(p(apa(sources[key]),'proposal_ref'))
    return story


def build() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'artifacts/results_summary.json').read_text())
    sources={s['id']:s for s in json.loads((ROOT/'docs/research/sources.json').read_text())['sources']}
    make_figures(summary)
    standalone=StudyPDF(OUT/'student_outcomes_research_proposal.pdf','Research proposal - Student outcomes')
    standalone.build(proposal_content(sources,True))

    source=sections(ROOT/'docs/report.md');story=[]
    story.extend([Spacer(1,37*mm),p('MACHINE LEARNING FINAL PROJECT','subtitle'),
                  p('Student Outcome Classification and Academic Profile Analysis','title'),
                  p('Three-class classification and academic profile analysis','subtitle'),
                  p('Team: ' + ', '.join(TEAM) + '.'),
                  p('UCI 697. Three educational outcomes. First-semester models and information profiles.'),Spacer(1,8*mm)])
    story.append(table(['Team member','Proposed technical scope'],[
        [TEAM[0],'ANN, training history and error analysis'],[TEAM[1],'Data preparation, EDA, K-Means and PCA'],
        [TEAM[2],'Classical models, CV and tuning']],[1,2]))
    story.extend([Spacer(1,6*mm),p('The roles are proposed. Actual technical contributions require separate records. AI assistance is disclosed in this report.','caption'),
                  p('19 November 2026. Draft for team review.','caption'),PageBreak(),heading('Abstract','abstract')])
    story.extend(paragraphs(source['Abstract']));story.append(PageBreak())
    story.append(heading('1 Introduction and related work','introduction'));story.extend(paragraphs(source['Introduction']))
    literature=sections(ROOT/'docs/literature_review.md')
    for title,body in literature.items():
        if title=='Position of this project':continue
        body=body.replace('Full reference entries appear in the [bibliography](bibliography.md).','')
        story.append(heading(title,level=2));story.extend(paragraphs(body))

    story.append(heading('2 Dataset and exploratory analysis','dataset'));story.extend(paragraphs(source['Dataset and EDA']))
    story.append(table(['Outcome','Entire dataset','Share'],[[c,str(n),f'{n/4424:.2%}'] for c,n in [('Dropout',1421),('Enrolled',794),('Graduate',2209)]],[1.5,1,1]))
    captions=[('eda_classes','Figure 1. Training class counts: 994 Dropout, 556 Enrolled and 1,546 Graduate. Source: saved split manifest.'),
              ('eda_age','Figure 2. Age by outcome in training data. The distributions overlap and valid extreme values are retained.'),
              ('eda_passed','Figure 3. Passed first-semester units in training data. Means are 2.58, 4.40 and 6.17 for Dropout, Enrolled and Graduate.'),
              ('eda_correlations','Figure 4. Selected quantitative training correlations. Passed units and mean grade have Pearson r=0.702.'),
              ('eda_courses','Figure 5. Training outcomes by programme. Labels show programme codes and group sizes. Small groups require caution.')]
    for name,caption in captions:story.append(figure(name,caption,maximum_height=360 if name=='eda_courses' else 290))

    story.append(heading('3 Methodology','methodology'));story.extend(paragraphs(source['Methodology']))
    split=pd.read_csv(ROOT/'artifacts/tables/split_counts.csv')
    story.append(table(['Partition','Dropout','Enrolled','Graduate','Total'],[[r['split'],r.Dropout,r.Enrolled,r.Graduate,r.Dropout+r.Enrolled+r.Graduate] for _,r in split.iterrows()],[1.5,1,1,1,1]))
    story.append(Spacer(1,8));story.append(table(['Algorithm','Selected parameters','CV Macro F1'],[
        ['Logistic Regression','C=0.1, class_weight=balanced','0.6483 ± 0.0052'],
        ['Decision Tree','max_depth=3, min_samples_leaf=5, balanced','0.6112 ± 0.0222'],
        ['KNN','n_neighbors=15, uniform','0.5612 ± 0.0184']],[1.3,2.5,1.2]))
    story.append(p('CV is reported as mean ± standard deviation across five folds within the training partition.','caption'))

    story.append(heading('4 Main results and ANN','results'));story.extend(paragraphs(source['Model results and ANN']))
    comparison=pd.read_csv(ROOT/'artifacts/tables/final_comparison.csv')
    story.append(table(['Model','Validation Macro F1','Test Macro F1','Test accuracy'],[[r.model,f'{r.validation_macro_f1:.4f}',f'{r.test_macro_f1:.4f}',f'{r.test_accuracy:.2%}'] for _,r in comparison.iterrows()],[1.7,1,1,1]))
    story.append(p('Validation selects ANN. Source: final_comparison.csv. Every model is fitted only on training data.','caption'))
    story.append(figure('main_comparison','Figure 6. Main comparison with a shared first-semester profile and test set. The different test ranking does not revise selection.'))
    story.append(figure('ann_history','Figure 7. Recorded history of the selected ANN: checkpoint 9, stopping epoch 34. Loss curves show unweighted cross-entropy.',maximum_height=350))

    story.append(heading('5 Effect of semester information','ablation'));story.extend(paragraphs(source['Feature-group comparison']))
    ablation=pd.read_csv(ROOT/'artifacts/tables/feature_ablation.csv')
    labels=['No semester results','First semester','Both semesters']
    story.append(table(['Profile','Features before one-hot','Validation Macro F1','Test Macro F1'],[[labels[i],r.input_features,f'{r.validation_macro_f1:.4f}',f'{r.test_macro_f1:.4f}'] for i,(_,r) in enumerate(ablation.iterrows())],[1.7,1,1,1]))
    story.append(figure('ablation','Figure 8. Logistic Regression with a shared grid and CV procedure. The two-semester profile uses additional information.'))

    story.append(heading('6 Clustering and PCA','unsupervised'));story.extend(paragraphs(source['Clustering and PCA']))
    selection=pd.read_csv(ROOT/'artifacts/tables/cluster_selection.csv')
    story.append(table(['k','Silhouette','Minimum cluster share','Mean seed ARI'],[[int(r.k),f'{r.silhouette:.4f}',f'{r.minimum_cluster_fraction:.2%}',f'{r.mean_seed_ari:.4f}'] for _,r in selection.iterrows()],[.5,1,1.4,1.1]))
    story.append(Spacer(1,8));story.append(table(['Cluster','n','Mean enrolled units','Mean approved units','Dropout share'],[
        [str(r['cluster']),str(r['n']),f"{r['mean_Curricular units 1st sem (enrolled)']:.2f}",f"{r['mean_Curricular units 1st sem (approved)']:.2f}",f"{r['share_Dropout']:.1%}"] for r in summary['unsupervised']['cluster_profiles']],[.7,.7,1.3,1.3,1.1]))
    story.append(p('Enrolled and approved refer to numbers of curricular units here. Target did not enter cluster selection.','caption'))
    story.append(figure('pca_variance','Figure 9. Explained variance of five standardised academic indicators. Four components retain 97.92%.'))
    story.append(figure('pca_clusters','Figure 10. Training observations projected onto the two saved principal components, coloured by cluster.'))
    story.append(figure('pca_classes','Figure 11. The same projection with outcomes added after cluster selection. Outcome classes overlap.'))

    story.append(heading('7 Error analysis and discussion','discussion'));story.extend(paragraphs(source['Error analysis and discussion']))
    matrix=pd.read_csv(ROOT/'artifacts/tables/selected_confusion_matrix.csv',index_col=0)
    story.append(table(['Actual / predicted','Dropout','Enrolled','Graduate'],[[c,*[str(int(matrix.loc[c,j])) for j in CLASSES]] for c in CLASSES],[1.7,1,1,1]))
    story.append(p('Selected ANN confusion matrix on 664 test observations. Rows are actual outcomes.','caption'))
    metrics=pd.read_csv(ROOT/'artifacts/tables/test_class_metrics.csv');metrics=metrics[metrics.model=='ANN']
    story.append(table(['Class','Precision','Recall','F1','n'],[[r['class'],f'{r.precision:.4f}',f'{r.recall:.4f}',f'{r.f1:.4f}',int(r.support)] for _,r in metrics.iterrows()],[1.3,1,1,1,.6]))

    story.append(PageBreak());story.append(heading('Conclusion','conclusion'));story.extend(paragraphs(source['Conclusion']))
    story.append(heading('AI assistance and team contributions',level=2));story.extend(paragraphs(source['AI assistance']))
    story.append(PageBreak());story.extend(proposal_content(sources,False))
    story.append(PageBreak());story.append(heading('Bibliography','bibliography'))
    bibliography=(ROOT/'docs/bibliography.md').read_text().split('\n\n')[2:]
    story.extend(p(item,'bibliography') for item in bibliography if item.strip())
    document=StudyPDF(OUT/'student_outcomes_report.pdf','Student Outcome Classification and Academic Profile Analysis')
    document.build(story)
    report_pages=len(PdfReader(OUT/'student_outcomes_report.pdf').pages)
    proposal_pages=len(PdfReader(OUT/'student_outcomes_research_proposal.pdf').pages)
    layout={'language':'en','report_pages':report_pages,'proposal_pages':proposal_pages,'section_start_pages':document.section_pages,
            'main_body_pages_before_conclusion':document.section_pages['conclusion']-document.section_pages['introduction'],
            'status':'generated_visual_review_pending'}
    write_json(ROOT/'artifacts/report_layout.json',layout)
    print(json.dumps(layout,ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
