"""Springer LNCS layout on top of the supplied conference template (conf_templates.docm).

The .docm is converted ONCE to a macro-free .docx (VBA project, ribbon UI and sample chart removed; nothing is executed).
All styles (papertitle, author, address, abstract, keywords, heading1/2, p1a, tablecaption, figurecaption, equation,
referenceitem, ...) come from the template itself, so numbering, fonts, running heads and page geometry are the template's.
"""
from __future__ import annotations

import os
import re
import shutil
import zipfile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# ---------------------------------------------------------------------------------------------- template cleaning
def make_clean_template(docm: str, out_docx: str) -> str:
    """Strip macros / ribbon UI / sample chart from the .docm and write a plain .docx with the same styles."""
    drop_prefix = ("word/vbaProject.bin", "word/vbaData.xml", "word/_rels/vbaProject.bin.rels", "customUI/",
                   "word/charts/")
    with zipfile.ZipFile(docm) as zin, zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name.startswith(drop_prefix):
                continue
            data = zin.read(name)
            if name == "[Content_Types].xml":
                t = data.decode("utf8")
                t = t.replace("application/vnd.ms-word.document.macroEnabled.main+xml",
                              "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml")
                t = re.sub(r'<Override PartName="/word/(vbaData\.xml|charts/[^"]*)"[^>]*/>', "", t)
                data = t.encode("utf8")
            elif name == "word/_rels/document.xml.rels":
                t = data.decode("utf8")
                t = re.sub(r'<Relationship [^>]*(vbaProject|vbaData|relationships/chart)[^>]*/>', "", t)
                data = t.encode("utf8")
            elif name == "_rels/.rels":
                t = data.decode("utf8")
                t = re.sub(r'<Relationship [^>]*customUI[^>]*/>', "", t)
                data = t.encode("utf8")
            elif name.startswith("word/header") and name.endswith(".xml"):
                t = data.decode("utf8")
                t = t.replace("F. Author and S. Author", "P. Balakrishnan and A. Anny Leema")
                t = t.replace("Contribution Title (shortened if too long)", "Agentic Scout v2")
                data = t.encode("utf8")
            zout.writestr(item, data)
    return out_docx


# ---------------------------------------------------------------------------------------------- LNCS-format references
LNCS_REFS = {  # key -> reference text in Springer LNCS style (VERIFY every entry before submission)
 "abouali": "Abou Ali, M., Dornaika, F.: Agentic AI: a comprehensive survey of architectures, applications, and future directions. arXiv:2510.25445 (2025)",
 "wang24": "Wang, L., et al.: A survey on large language model based autonomous agents. Front. Comput. Sci. 18, 186345 (2024)",
 "sapkota": "Sapkota, R., Roumeliotis, K.I., Karkee, M.: AI agents vs. agentic AI: a conceptual taxonomy, applications and challenges. Inf. Fusion, 103599 (2025)",
 "hong24": "Hong, S., et al.: Data Interpreter: an LLM agent for data science. arXiv:2402.18679 (2024)",
 "hollmann23": "Hollmann, N., Müller, S., Hutter, F.: Large language models for automated data science: introducing CAAFE for context-aware automated feature engineering. In: Advances in Neural Information Processing Systems, vol. 36 (2023)",
 "he21": "He, X., Zhao, K., Chu, X.: AutoML: a survey of the state-of-the-art. Knowl.-Based Syst. 212, 106622 (2021)",
 "feurer15": "Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., Hutter, F.: Efficient and robust automated machine learning. In: Advances in Neural Information Processing Systems, vol. 28 (2015)",
 "feurer15b": "Feurer, M., Springenberg, J.T., Hutter, F.: Initializing Bayesian hyperparameter optimization via meta-learning. In: Proceedings of the 29th AAAI Conference on Artificial Intelligence, pp. 1128–1135 (2015)",
 "thornton13": "Thornton, C., Hutter, F., Hoos, H.H., Leyton-Brown, K.: Auto-WEKA: combined selection and hyperparameter optimization of classification algorithms. In: Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 847–855 (2013)",
 "olson16": "Olson, R.S., Moore, J.H.: TPOT: a tree-based pipeline optimization tool for automating machine learning. In: Proceedings of the Workshop on Automatic Machine Learning, PMLR, vol. 64, pp. 66–74 (2016)",
 "ledell20": "LeDell, E., Poirier, S.: H2O AutoML: scalable automatic machine learning. In: 7th ICML Workshop on Automated Machine Learning (2020)",
 "wang21": "Wang, C., Wu, Q., Weimer, M., Zhu, E.: FLAML: a fast and lightweight AutoML library. In: Proceedings of Machine Learning and Systems (MLSys) (2021)",
 "erickson20": "Erickson, N., et al.: AutoGluon-Tabular: robust and accurate AutoML for structured data. arXiv:2003.06505 (2020)",
 "gijsbers24": "Gijsbers, P., et al.: AMLB: an AutoML benchmark. J. Mach. Learn. Res. 25(101), 1–65 (2024)",
 "romano21": "Romano, J.D., et al.: PMLB v1.0: an open-source dataset collection for benchmarking machine learning methods. Bioinformatics 38(3), 878–880 (2022)",
 "bischl21": "Bischl, B., et al.: OpenML benchmarking suites. In: NeurIPS Datasets and Benchmarks Track (2021)",
 "vanschoren13": "Vanschoren, J., van Rijn, J.N., Bischl, B., Torgo, L.: OpenML: networked science in machine learning. SIGKDD Explor. 15(2), 49–60 (2013)",
 "vanschoren18": "Vanschoren, J.: Meta-learning: a survey. arXiv:1810.03548 (2018)",
 "brazdil09": "Brazdil, P., Giraud-Carrier, C., Soares, C., Vilalta, R.: Metalearning: Applications to Data Mining. Springer, Heidelberg (2009)",
 "mccloskey89": "McCloskey, M., Cohen, N.J.: Catastrophic interference in connectionist networks: the sequential learning problem. Psychol. Learn. Motiv. 24, 109–165 (1989)",
 "kirkpatrick17": "Kirkpatrick, J., et al.: Overcoming catastrophic forgetting in neural networks. Proc. Natl. Acad. Sci. USA 114(13), 3521–3526 (2017)",
 "rebuffi17": "Rebuffi, S.A., Kolesnikov, A., Sperl, G., Lampert, C.H.: iCaRL: incremental classifier and representation learning. In: IEEE Conference on Computer Vision and Pattern Recognition, pp. 5533–5542 (2017)",
 "lopezpaz17": "Lopez-Paz, D., Ranzato, M.: Gradient episodic memory for continual learning. In: Advances in Neural Information Processing Systems, vol. 30 (2017)",
 "chaudhry18": "Chaudhry, A., Dokania, P.K., Ajanthan, T., Torr, P.H.S.: Riemannian walk for incremental learning: understanding forgetting and intransigence. In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds.) ECCV 2018. LNCS, vol. 11215, pp. 532–547. Springer, Cham (2018)",
 "vandeven19": "van de Ven, G.M., Tolias, A.S.: Three scenarios for continual learning. arXiv:1904.07734 (2019)",
 "parisi19": "Parisi, G.I., Kemker, R., Part, J.L., Kanan, C., Wermter, S.: Continual lifelong learning with neural networks: a review. Neural Netw. 113, 54–71 (2019)",
 "delange22": "De Lange, M., et al.: A continual learning survey: defying forgetting in classification tasks. IEEE Trans. Pattern Anal. Mach. Intell. 44(7), 3366–3385 (2022)",
 "adadi18": "Adadi, A., Berrada, M.: Peeking inside the black-box: a survey on explainable artificial intelligence (XAI). IEEE Access 6, 52138–52160 (2018)",
 "lipton16": "Lipton, Z.C.: The mythos of model interpretability. arXiv:1606.03490 (2016)",
 "lundberg17": "Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems, vol. 30 (2017)",
 "frazzetto19": "Frazzetto, D., Nielsen, T.D., Pedersen, T.B., Šikšnys, L.: Prescriptive analytics: a survey of emerging trends and technologies. VLDB J. 28, 575–595 (2019)",
 "bertsimas20": "Bertsimas, D., Kallus, N.: From predictive to prescriptive analytics. Manage. Sci. 66(3), 1025–1044 (2020)",
 "chen12": "Chen, H., Chiang, R.H.L., Storey, V.C.: Business intelligence and analytics: from big data to big impact. MIS Q. 36(4), 1165–1188 (2012)",
 "tukey77": "Tukey, J.W.: Exploratory Data Analysis. Addison-Wesley, Reading (1977)",
 "mukhiya20": "Mukhiya, S.K., Ahmed, U.: Hands-On Exploratory Data Analysis with Python. Packt Publishing, Birmingham (2020)",
 "little88": "Little, R.J.A.: A test of missing completely at random for multivariate data with missing values. J. Am. Stat. Assoc. 83(404), 1198–1202 (1988)",
 "vanbuuren11": "van Buuren, S., Groothuis-Oudshoorn, K.: mice: multivariate imputation by chained equations in R. J. Stat. Softw. 45(3), 1–67 (2011)",
 "liu08": "Liu, F.T., Ting, K.M., Zhou, Z.H.: Isolation forest. In: 8th IEEE International Conference on Data Mining, pp. 413–422 (2008)",
 "saito15": "Saito, T., Rehmsmeier, M.: The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE 10(3), e0118432 (2015)",
 "brodersen10": "Brodersen, K.H., Ong, C.S., Stephan, K.E., Buhmann, J.M.: The balanced accuracy and its posterior distribution. In: 20th International Conference on Pattern Recognition, pp. 3121–3124 (2010)",
 "demsar06": "Demšar, J.: Statistical comparisons of classifiers over multiple data sets. J. Mach. Learn. Res. 7, 1–30 (2006)",
 "pedregosa11": "Pedregosa, F., et al.: Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12, 2825–2830 (2011)",
 "grinsztajn22": "Grinsztajn, L., Oyallon, E., Varoquaux, G.: Why do tree-based models still outperform deep learning on typical tabular data? In: NeurIPS Datasets and Benchmarks Track (2022)",
 "gama14": "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A.: A survey on concept drift adaptation. ACM Comput. Surv. 46(4), 44 (2014)",
 "cattell63": "Cattell, R.B.: Theory of fluid and crystallized intelligence: a critical experiment. J. Educ. Psychol. 54(1), 1–22 (1963)",
 "lewis20": "Lewis, P., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems, vol. 33 (2020)",
 "finn17": "Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: Proceedings of the 34th International Conference on Machine Learning, pp. 1126–1135 (2017)",
 "auer02": "Auer, P., Cesa-Bianchi, N., Fischer, P.: Finite-time analysis of the multiarmed bandit problem. Mach. Learn. 47, 235–256 (2002)",
 "sutton18": "Sutton, R.S., Barto, A.G.: Reinforcement Learning: An Introduction, 2nd edn. MIT Press, Cambridge (2018)",
 "feurer22": "Feurer, M., Eggensperger, K., Falkner, S., Lindauer, M., Hutter, F.: Auto-sklearn 2.0: hands-free AutoML via meta-learning. J. Mach. Learn. Res. 23 (2022)",
 "fusi18": "Fusi, N., Sheth, R., Elibol, M.: Probabilistic matrix factorization for automated machine learning. In: Advances in Neural Information Processing Systems, pp. 3348–3357 (2018)",
 "yang19": "Yang, C., Akimoto, Y., Kim, D.W., Udell, M.: OBoe: collaborative filtering for AutoML model selection. In: KDD 2019, pp. 1173–1183. ACM (2019)",
 "alcobaca20": "Alcobaça, E., Siqueira, F., Rivolli, A., Garcia, L.P.F., Oliva, J.T., de Carvalho, A.C.P.L.F.: MFE: towards reproducible meta-feature extraction. J. Mach. Learn. Res. 21 (2020)",
 "maher19": "Maher, M., Sakr, S.: SmartML: a meta learning-based framework for automated selection and hyperparameter tuning for machine learning algorithms. In: EDBT 2019, pp. 554–557 (2019)",
 "garouani22": "Garouani, M., Ahmad, A., Bouneffa, M., Hamlich, M., Bourguin, G., Lewandowski, A.: Using meta-learning for automated algorithms selection and configuration: an experimental framework for industrial big data. J. Big Data 9(1) (2022)",
 "das20": "Das, P., et al.: Amazon SageMaker Autopilot: a white box AutoML solution at scale. In: DEEM 2020. ACM (2020)",
 "wangma23": "Wang, Z., Su, K., Zhang, J., Jia, H., Ye, Q., Xie, X., Lu, Z.: Multi-agent automated machine learning. In: CVPR 2023, pp. 11960–11969. IEEE (2023)",
 "trirat25": "Trirat, P., Jeong, W., Hwang, S.J.: AutoML-Agent: a multi-agent LLM framework for full-pipeline AutoML. Proc. Mach. Learn. Res. 267, 60099–60146 (2025)",
 "fang25": "Fang, H., et al.: MLZero: a multi-agent system for end-to-end machine learning automation. In: Advances in Neural Information Processing Systems, vol. 38 (2025)",
 "gandhi25": "Gandhi, S., Patwardhan, M., Vig, L., Shroff, G.: BudgetMLAgent: a cost-effective LLM multi-agent system for automating machine learning tasks. In: AIMLSystems 2024. ACM (2025)",
 "bakas25": "Bakas, N., Lavdas, S., Vavousis, K., Christodoulou, C., Langousis, A.: Automated machine learning in a multi-agent environment. LNBIP, vol. 535, pp. 47–57. Springer (2025)",
 "patel23": "Patel, H., Guttula, S., Gupta, N., Hans, S., Mittal, R.S., Lokesh, N.: A data-centric AI framework for automating exploratory data analysis and data quality tasks. J. Data Inf. Qual. 15(4) (2023)",
 "goyle24": "Goyle, K., Xie, Q., Goyle, V.: DataAssist: a machine learning approach to data cleaning and preparation. LNNS, vol. 1067, pp. 476–486. Springer (2024)",
 "turner24": "Turner, D., Cardoso, P.J.S., Rodrigues, J.M.F.: Continual learning for object classification: integrating AutoML for binary classification tasks within a modular dynamic architecture. IEEE Access 12, 183725–183742 (2024)",
 "ang25": "Ang, Y., Bao, Y., Jiang, L., Tao, J., Tung, A.K.H., Szpurch, L., Ni, H.: Structured agentic workflows for financial time-series modelling with LLMs and reflective feedback. In: ICAIF 2025, pp. 924–932. ACM (2025)",
 "vanrijn18": "van Rijn, J.N., Hutter, F.: Hyperparameter importance across datasets. In: KDD 2018, pp. 2367–2376. ACM (2018)",
 "vaccaro21": "Vaccaro, L., Sansonetti, G., Micarelli, A.: An empirical review of automated machine learning. Computers 10(1), 1–27 (2021)",
}


# ---------------------------------------------------------------------------------------------- document builder
class LNCS:
    TEXT_W = 6900  # twips (12.2 cm), the template's text width

    def __init__(self, template_docx: str):
        self.doc = Document(template_docx)
        body = self.doc.element.body
        sect = body.find(qn("w:sectPr"))
        for el in list(body):
            if el is not sect:
                body.remove(el)
        self.cite_order, self.todo = [], 0
        self.first = True
        self.tn = self.fn = self.en = 0

    # -- helpers
    def _style(self, p, sid):
        p.style = self.doc.styles.get_by_id(sid, WD_STYLE_TYPE.PARAGRAPH)
        return p

    def _cites(self, text):
        def sub(m):
            nums = []
            for k in m.group(1).split(","):
                k = k.strip().lstrip("@")
                if k not in LNCS_REFS:
                    raise KeyError(k)
                if k not in self.cite_order:
                    self.cite_order.append(k)
                nums.append(self.cite_order.index(k) + 1)
            return "[" + ",".join(map(str, sorted(nums))) + "]"
        return re.sub(r"\[((?:@\w+,?\s*)+)\]", sub, text)

    def _runs(self, p, text, bold=False, italic=False, size=None):
        for part in re.split(r"(\*\*.+?\*\*|\[\[TODO:.+?\]\])", text):
            if not part:
                continue
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("[[TODO"):
                r = p.add_run(part[2:-2]); r.bold = True; r.font.color.rgb = RGBColor(0xC0, 0, 0); self.todo += 1
            else:
                r = p.add_run(part); r.bold = bold or None
            if italic:
                r.italic = True
            if size:
                r.font.size = Pt(size)

    # -- front matter
    def front(self, title, authors, affiliation_lines, banner=None):
        self._style(self.doc.add_paragraph(), "papertitle").add_run(title)
        p = self._style(self.doc.add_paragraph(), "author")
        for i, a in enumerate(authors):
            star = a.endswith("*")
            p.add_run((" and " if i == len(authors) - 1 and i > 0 else (", " if i else "")) + a.rstrip("*"))
            p.add_run("1" + ("*" if star else "")).font.superscript = True
        for line in affiliation_lines:
            self._runs(self._style(self.doc.add_paragraph(), "address"), line)
        if banner:
            q = self._style(self.doc.add_paragraph(), "address")
            r = q.add_run(banner); r.bold = True; r.font.color.rgb = RGBColor(0xC0, 0, 0)

    def abstract(self, text, keywords):
        n = len(text.split())
        if n > 250:
            print(f"WARNING: abstract has {n} words (LNCS limit 250)")
        p = self._style(self.doc.add_paragraph(), "abstract")
        r = p.add_run("Abstract. "); r.bold = True
        self._runs(p, text)
        k = self._style(self.doc.add_paragraph(), "keywords")
        r = k.add_run("Keywords: "); r.bold = True
        k.add_run(keywords)
        self.abstract_words = n

    # -- headings (numbering comes from the template styles)
    def h1(self, text):
        text = re.sub(r"^[IVX]+\.\s+", "", text)
        p = self._style(self.doc.add_paragraph(), "heading1")
        p.add_run(text)
        self.first = True
        return p

    def h2(self, text):
        text = re.sub(r"^[A-Z]\.\s+", "", text)
        self._style(self.doc.add_paragraph(), "heading2").add_run(text)
        self.first = True

    # -- body
    def p(self, text):
        p = self.doc.add_paragraph()
        if self.first:
            self._style(p, "p1a")
        self._runs(p, self._cites(text))
        self.first = False

    def bullets(self, items):
        for it in items:
            self._runs(self._style(self.doc.add_paragraph(), "bulletitem"), self._cites(it))
        self.first = True

    def eq(self, text, num):
        p = self._style(self.doc.add_paragraph(), "equation")
        r = p.add_run(); r.add_tab()
        r = p.add_run(text); r.italic = True
        r = p.add_run(); r.add_tab()
        p.add_run(f"({num})")
        self.first = True

    def two_columns(self):  # LNCS is single-column; kept for API compatibility
        pass

    def table(self, df, caption, widths=None, size=8.5):
        self.tn += 1
        caption = re.sub(r"^TABLE [IVX]+(-b)?\.\s*", "", caption)
        cp = self._style(self.doc.add_paragraph(), "tablecaption")
        cp.paragraph_format.keep_with_next = True
        r = cp.add_run(f"Table {self.tn}. "); r.bold = True
        cp.add_run(caption)
        n = len(df.columns)
        def colw(i, c):
            cells = [str(c)] + [str(v) for v in df.iloc[:, i]]
            longest_word = max(len(wd) for x in cells for wd in x.split()) if cells else 5
            return max(longest_word * 1.15, 0.35 * max(len(x) for x in cells), 5)
        w = widths or [colw(i, c) for i, c in enumerate(df.columns)]
        tot = sum(w)
        cw = [int(self.TEXT_W * x / tot) for x in w]
        t = self.doc.add_table(rows=0, cols=n)
        for ci_, col_ in enumerate(t.columns):                 # update the grid so Word and LibreOffice both honour the widths
            col_.width = Inches(cw[ci_] / 1440)
        tblPr = t._tbl.tblPr
        for tag, attrs in (("w:tblW", {"w:w": str(sum(cw)), "w:type": "dxa"}), ("w:jc", {"w:val": "center"}),
                           ("w:tblLayout", {"w:type": "fixed"})):
            for old in tblPr.findall(qn(tag)):
                tblPr.remove(old)
            el = OxmlElement(tag)
            for k, v in attrs.items():
                el.set(qn(k), v)
            tblPr.append(el)
        rows = [list(df.columns)] + df.astype(str).values.tolist()
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for ci, val in enumerate(row):
                c = cells[ci]
                c.width = Inches(cw[ci] / 1440)
                tcPr = c._tc.get_or_add_tcPr()
                b = OxmlElement("w:tcBorders")
                edges = []
                if ri == 0:
                    edges += [("top", 12), ("bottom", 6)]
                if ri == len(rows) - 1:
                    edges += [("bottom", 12)]
                for edge, sz in edges:
                    e = OxmlElement(f"w:{edge}")
                    for k, v in (("w:val", "single"), ("w:sz", str(sz)), ("w:space", "0"), ("w:color", "000000")):
                        e.set(qn(k), v)
                    b.append(e)
                tcPr.append(b)
                para = c.paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                para.paragraph_format.first_line_indent = 0
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.keep_with_next = True      # keep a table on one page (caption stays with it too)
                run = para.add_run(str(val)); run.font.size = Pt(size); run.bold = (ri == 0) or None
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2)
        sp.paragraph_format.first_line_indent = 0
        for r_ in sp.runs:
            r_.font.size = Pt(4)
        self.first = True

    def figure(self, path, caption, width=4.6):
        caption = re.sub(r"^Fig\.\s*\d+\.\s*", "", caption)
        if not os.path.exists(path):
            self.p(f"[[TODO: figure missing: {path}]]")
            return
        self.fn += 1
        p = self._style(self.doc.add_paragraph(), "image")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(width))
        c = self._style(self.doc.add_paragraph(), "figurecaption")
        r = c.add_run(f"Fig. {self.fn}. "); r.bold = True
        c.add_run(caption)
        self.first = True

    def acknowledgments(self, text):
        p = self._style(self.doc.add_paragraph(), "acknowlegments")
        r = p.add_run("Disclosure of Interests. "); r.bold = True
        self._runs(p, text)

    def references(self):
        p = self.h1("References")
        pPr = p._p.get_or_add_pPr()                      # unnumbered "References" heading
        numPr = OxmlElement("w:numPr")
        for tag, val in (("w:ilvl", "0"), ("w:numId", "0")):
            e = OxmlElement(tag); e.set(qn("w:val"), val); numPr.append(e)
        pPr.append(numPr)
        for k in self.cite_order:
            para = self._style(self.doc.add_paragraph(), "referenceitem")
            rr = para.add_run(LNCS_REFS[k]); rr.font.size = Pt(8)
            para.paragraph_format.line_spacing = Pt(9.5)

    def save(self, path, title="", author=""):
        cp = self.doc.core_properties
        cp.title, cp.author = title, author
        self.doc.save(path)
