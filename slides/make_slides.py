import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
BG, BG2, DARK, INK, MUTED, ACC, ACC2, CARD, LINE = "#f7f5f0", "#eeeae1", "#14232d", "#1f2933", "#56626b", "#0b6e99", "#b8452d", "#ffffff", "#ddd8cc"
HEAD = "font-family:'Source Serif 4', Georgia, serif"
BODY = "font-family:'Inter', Helvetica, Arial, sans-serif"
IMG = dict(classes="790e1319b1de715453b6d57c9afe1738", one="f9cf48b47c8b12e31ceb081c4f835ea8", protocol="90062b1e6f2abab4b4700585160f38e4",
           arch="14ae1bf3de0ac06aacc11b34cd2c02bd", archs="06c8287e7229c6dae5911c08963f3f05", leak="e55f8be4d3429e88ebd92a3726e7e77f",
           optlr="6609f7b6c8c0cad163b98f0d7224ba63", conf="d7970ba680f455d4c403d7ca4402d5d4", writers="ec90f7e28b5338c9c81f55b7c6cbe9b8",
           words="398a1b19e19a50c339aea1f8a9bedacb")
slides = []

def img(k, w, h, alt):
    return f'<img src="/_blob/{IMG[k]}" alt="{alt}" style="width:{w}px;height:{h}px;object-fit:contain;background:#ffffff;border:1px solid {LINE};border-radius:12px">'

def card(title, body, flex="1", accent=None):
    top = f"border-top:6px solid {accent}" if accent else f"border-top:1px solid {LINE}"
    return (f'<div style="flex:{flex};display:flex;flex-direction:column;gap:12px;background:{CARD};padding:28px 32px;border:1px solid {LINE};{top};border-radius:12px">'
            f'<h3 style="font-size:30px;font-weight:600;line-height:1.2;color:{INK}">{title}</h3>{body}</div>')

def p(text, size=26, color=INK, extra=""):
    return f'<p style="font-size:{size}px;line-height:1.35;color:{color};{extra}">{text}</p>'

def ul(items, size=26):
    return f'<ul style="font-size:{size}px;line-height:1.35;color:{INK}">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

def stat(num, label, color=ACC, flex="1"):
    return (f'<div style="flex:{flex};display:flex;flex-direction:column;gap:6px;background:{CARD};padding:24px 28px;border:1px solid {LINE};border-radius:12px">'
            f'<p style="{HEAD};font-size:64px;font-weight:600;line-height:1.05;color:{color}">{num}</p>'
            f'<p style="font-size:24px;line-height:1.3;color:{MUTED}">{label}</p></div>')

def slide(sid, n, title, body, notes, kicker=None, bg=BG):
    k = f'<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{ACC2}">{kicker}</p>' if kicker else ""
    slides.append((sid, f'''<section id="{sid}" data-transition="fade" style="background:{bg};color:{INK};{BODY};padding:88px 128px 144px;display:flex;flex-direction:column;gap:28px">
<div style="display:flex;flex-direction:column;gap:8px">{k}<h2 style="{HEAD};font-size:60px;font-weight:600;line-height:1.1;color:{INK}">{title}</h2></div>
{body}
<p style="position:absolute;left:128px;bottom:64px;width:1200px;font-size:24px;color:{MUTED}">Aksara Jawa recognition · Group mini project · Introduction to Deep Learning</p>
<p style="position:absolute;right:128px;bottom:64px;width:120px;text-align:right;font-size:24px;color:{MUTED}">{n} / 12</p>
<aside>{notes}</aside>
</section>'''))

# 1 cover -----------------------------------------------------------------
members = ["Syakirah", "Balma", "Nabila", "Aditya Wahyu"]
mc = "".join(f'<div style="flex:1;display:flex;flex-direction:column;gap:6px;padding:24px 28px;border:1px solid #3b5261;border-radius:12px;background:#1c303c">'
             f'<p style="font-size:32px;font-weight:600;color:#f7f5f0">{m}</p><p style="font-size:24px;color:#b9c6ce">[NIM]</p></div>' for m in members)
slides.append(("cover", f'''<section id="cover" data-transition="fade" style="background:{DARK};color:#f7f5f0;{BODY};padding:128px;display:flex;flex-direction:column;justify-content:space-between;gap:40px">
<div style="display:flex;flex-direction:column;gap:28px">
<p style="font-size:26px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:#e9b44c">Group mini project · Introduction to Deep Learning</p>
<h1 style="{HEAD};font-size:104px;font-weight:600;line-height:1.05;color:#f7f5f0">Reading Aksara Jawa with deep learning</h1>
<p style="font-size:36px;line-height:1.35;color:#d5dde2;width:1400px">Handwritten Javanese script recognition, evaluated without data leakage</p>
</div>
<div style="display:flex;flex-direction:column;gap:20px">
<div style="display:flex;gap:24px">{mc}</div>
<p style="font-size:24px;color:#b9c6ce">Master of Computer Science, Universitas Gadjah Mada · Lecturer: Afiahayati · 16 October 2026</p>
</div>
<aside>Opening. We built a recogniser for the 20 base characters of the Javanese script and, along the way, found that how you split this dataset matters as much as the model. Introduce the four members.</aside>
</section>'''))

# 2 jobdesc ---------------------------------------------------------------
jobs = [("Aditya Wahyu", "Data and protocol", ["Dataset audit and leakage analysis", "5-fold protocol grouped by drawing", "Experiment pipeline and code repository"]),
        ("Syakirah", "Architectures", ["Baselines: linear, MLP, LeNet-5, lecture CNN, Kaggle CNN", "Design of AksaraNet and ResNet", "Experiment 1: architecture comparison"]),
        ("Balma", "Learning strategy", ["Optimiser × learning-rate grid", "Regularisation, augmentation, schedule, capacity", "Experiment 2: tuning and final recipe"]),
        ("Nabila", "Evaluation and application", ["Final model, error analysis, unseen-writer test", "Word reading on photographs", "Slides and presentation"])]
body = '<div style="display:flex;gap:24px;flex:1">' + "".join(
    card(n, p(r, 26, ACC2, "font-weight:600") + ul(b, 26), accent=ACC) for n, r, b in jobs) + "</div>"
slide("jobdesc", 2, "Who did what", body, "Each member owned one part end to end; everyone reviewed the final experiments and the slides together.", "Job description")

# 3 problem ---------------------------------------------------------------
body = ('<div style="display:flex;gap:24px">'
        + card("The script", p("Aksara Jawa (Hanacaraka) has 20 base characters. Few people still read it fluently, and most manuscripts are not digitised."))
        + card("The task", p("Classify an image of one handwritten character into 20 classes, then read whole handwritten words into Latin script."))
        + card("Why it is hard", p("Glyphs differ by one small stroke (na / ka, ha / la), handwriting varies between writers, and labelled data is scarce."))
        + '</div>' + img("classes", 1664, 400, "The 20 Javanese base characters, each shown in six offline variants: original, rotated by plus and minus 5 and 10 degrees, and bold"))
slide("problem", 3, "Recognising handwritten Javanese characters", body,
      "Problem case. Baseline to beat: the public Kaggle notebook that ships with the dataset, a four-layer CNN reporting 96.5 percent test accuracy. The figure shows the 20 classes and the six variants stored for each drawing.", "Problem case")

# 4 data ------------------------------------------------------------------
left = ('<div style="flex:1;display:flex;flex-direction:column;gap:20px">'
        '<div style="display:flex;gap:20px">' + stat("12,000", "images in 20 balanced classes (Kaggle, secondary data)") + stat("420", "drawings behind them, ~29 augmented copies each", ACC2) + '</div>'
        + ul(["<b>Shipped split:</b> the test folder has only 42 drawings, mostly by one writer",
              "<b>Ours:</b> 5-fold cross-validation <b>grouped by drawing</b>",
              "Per rotation: 3 folds train, 1 validation (all decisions), 1 test (report only)"], 26) + '</div>')
right = '<div style="display:flex;flex-direction:column;gap:20px">' + img("protocol", 760, 268, "Diagram of the five rotations: each fold of about 84 drawings is the test set once, the next fold is validation, the other three are training") + \
        p("A random image-level split would put copies of the same drawing in both train and test.", 24, MUTED, "width:760px") + '</div>'
body = f'<div style="display:flex;gap:40px">{left}{right}</div>' + img("one", 1664, 236, "All 32 images derived from one single drawing of the character nya: rotated, bold and randomly augmented copies")
slide("data", 4, "12,000 images, but only 420 drawings", body,
      "The file names encode class, sample id, writer and augmentation. Parsing them shows every image descends from one of 420 drawings. The bottom strip shows all 32 images of a single drawing. Copies are not independent samples, so we split by drawing. 5 folds of about 84 drawings; mean and standard deviation over folds; confidence intervals by bootstrap over drawings.", "Data and splitting")

# 5 architecture ----------------------------------------------------------
body = (img("arch", 1664, 262, "AksaraNet architecture: input 1 by 64 by 64, four stages of two 3 by 3 convolutions with batch normalisation and ReLU with 16, 32, 64 and 128 filters, max pooling between stages, global average pooling, dropout 0.3, dense layer with 20 outputs")
        + '<div style="display:flex;gap:24px">'
        + card("VGG-style stages", p("Two 3×3 convolutions per stage with padding: a 5×5 receptive field, fewer weights, strokes at the border are kept."), accent=ACC)
        + card("Batch normalisation", p("After every convolution. Without it the same network reaches 48 % instead of 99 % in the same training budget."), accent=ACC)
        + card("Global average pooling", p("Replaces Flatten + Dense(256). The head shrinks from 530k to 2.6k weights, which removes the main source of over-fitting."), accent=ACC)
        + '</div>'
        + p("<b>Compared against:</b> linear classifier, MLP, HOG + SVM, LeNet-5, the lecture CNN, the reference Kaggle CNN (2.08 M parameters) and a small ResNet. AksaraNet has <b>296 k parameters</b>, 7× fewer than the reference.", 26))
slide("architecture", 5, "AksaraNet: a small VGG-style CNN", body,
      "Architecture. Input is grayscale 64 by 64, inverted so ink is one. Softmax output over 20 classes. The comparison models follow the course: linear classifier, multi-layer perceptron, conventional feature extraction, LeNet, the CNN from the lecture slides, and ResNet.", "Architecture")

# 6 learning strategy -----------------------------------------------------
rows = [("Optimiser", "Adam", "AdamW, weight decay 0.01"), ("Learning rate", "Constant 0.001", "One-cycle, peak 0.003"),
        ("Augmentation", "None", "Rotation, scale, shear, shift, stroke width"), ("Label smoothing", "None", "0.1"),
        ("Epochs", "Up to 25, early stopping", "25, best validation epoch kept"),
        ("Unchanged", "Batch 64, dropout 0.3", "Batch 64, dropout 0.3")]
table = (f'<table style="font-size:26px;color:{INK};background:{CARD};border:1px solid {LINE};border-radius:12px">'
         + f'<tr style="background:{BG2}"><th style="width:24%;text-align:left">Setting</th><th style="width:33%;text-align:left">Baseline training</th><th style="width:43%;text-align:left">Tuned training</th></tr>'
         + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
         + f'<tr style="background:{BG2}"><td><b>Test accuracy</b></td><td><b>98.5 %</b></td><td><b><span style="color:{ACC2}">99.4 %</span></b></td></tr></table>')
left = f'<div style="flex:1;display:flex;flex-direction:column;gap:18px">{table}' + p(
    "<b>Baseline</b> = plain setup used to compare architectures. <b>Tuned</b> = what tuning on validation folds led to (about 100 runs); it trains the final model.", 24, MUTED) + '</div>'
right = '<div style="display:flex;flex-direction:column;gap:16px">' + img("optlr", 800, 360, "Heatmap of validation accuracy for five optimisers and four learning rates. SGD only works at 0.1; Adam, AdamW and RMSprop are near their best from 0.0001 to 0.01") + \
        p("Plain SGD only works at lr 0.1. Adam, AdamW and RMSprop stay within about one point of their best across three orders of magnitude.", 24, MUTED, "width:800px") + '</div>'
slide("strategy", 6, "Learning strategy and parameter tuning", f'<div style="display:flex;gap:40px">{left}{right}</div>',
      "Learning strategy. Two training setups for the same network. Baseline training is deliberately plain so that Experiment 1 compares architectures fairly. Tuned training is the result of Experiment 2: one factor at a time on validation folds, then the best settings combined. The test folds never choose anything. The heatmap is validation accuracy after at most 12 epochs: under-training, not over-fitting, is what hurts SGD with a small learning rate.", "Learning strategy")

# 7 results 1 -------------------------------------------------------------
body = ('<div style="display:flex;gap:32px">'
        + '<div style="display:flex;flex-direction:column;gap:14px">' + img("archs", 1010, 433, "Bar chart of test accuracy under 5-fold cross-validation: linear 42.1, MLP 52.7, HOG plus SVM 74.5, LeNet-5 64.9, lecture CNN 64.7, Kaggle CNN with reference recipe 88.7, Kaggle CNN 86.6, ResNet 97.2, AksaraNet 98.5")
        + p("Baseline training for all networks (Adam, lr 0.001, early stopping). Dots are the five folds.", 24, MUTED, "width:1010px") + '</div>'
        + '<div style="flex:1;display:flex;flex-direction:column;gap:14px">' + img("leak", 622, 320, "Bar chart comparing a random image-level split with the drawing-level split: MLP 95.3 versus 52.7, lecture CNN 98.1 versus 64.7, AksaraNet 99.96 versus 98.5")
        + p("<b>Why the split matters:</b> with a random image-level split the MLP looks 43 points better than it is.", 26, INK) + '</div></div>'
        + '<div style="display:flex;gap:20px">' + stat("98.5 %", "AksaraNet, mean of 5 folds", ACC2) + stat("86.6 %", "reference Kaggle CNN, same protocol") + stat("74.5 %", "HOG + SVM beats LeNet-5 and the lecture CNN (65 %)") + '</div>')
slide("results1", 7, "Experiment 1: architecture decides most of the accuracy", body,
      "Results one. Pixel models and shallow CNNs with a big dense head memorise the training drawings: near 100 percent training accuracy, 42 to 65 percent on new drawings. Hand-crafted HOG features beat them because with 250 training drawings a good prior matters. Depth helps; batch normalisation plus global average pooling gives the jump from 87 to 98.5. The right chart is the leakage demonstration.", "Results 1 of 3")

# 8 results 2 -------------------------------------------------------------
rows = [("Optimiser × learning rate (20 settings)", "10.7 to 99.6", "Decisive"),
        ("Batch normalisation: off / on", "48.1 / 99.3", "Decisive"),
        ("Convolutions per stage: 1 / 2 / 3", "93.6 / 99.3 / 99.7", "Matters; 2 kept (3 costs 1.7× time)"),
        ("Width: ×0.5 / ×1 / ×2", "97.8 / 99.3 / 98.9", "×1 is enough"),
        ("Input size: 32 / 64 / 128 px", "98.2 / 99.3 / 99.4", "64 px kept (128 px costs far more)"),
        ("Batch size: 16 / 64 / 256", "99.5 / 99.3 / 99.1", "Within noise"),
        ("Dropout: 0 / 0.3 / 0.5", "99.4 / 99.3 / 99.1", "Within noise"),
        ("Augmentation: none / light / medium / strong", "99.3 / 99.7 / 99.4 / 99.5", "Small, consistent gain"),
        ("Tuned training vs baseline training", "100.0 vs 99.3", "Test folds: 99.3 vs 98.0")]
table = (f'<table style="font-size:26px;color:{INK};background:{CARD};border:1px solid {LINE};border-radius:12px">'
         + f'<tr style="background:{BG2}"><th style="width:42%;text-align:left">Factor varied</th><th style="width:24%;text-align:left">Validation accuracy (%)</th><th style="width:34%;text-align:left">Verdict</th></tr>'
         + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows) + "</table>")
body = table + p("Mean of two validation folds. Near 99 %, a difference of 0.2 points is about 5 validation images, so only the first five rows are clearly real effects.", 24, MUTED)
slide("results2", 8, "Experiment 2: what tuning changed, and what it did not", body,
      "Results two. Be honest about noise: accuracy is at the ceiling, so most regularisation choices cannot be told apart. The clear effects are optimiser and learning rate, batch normalisation and capacity. Combining AdamW, one-cycle and strong augmentation gave 100 percent validation accuracy on both tuning folds and 99.3 on their test folds.", "Results 2 of 3")

# 9 results 3 -------------------------------------------------------------
body = ('<div style="display:flex;gap:20px">'
        + stat("99.41 %", "final model, 5 folds × 3 seeds (sd 0.78)", ACC2)
        + stat("99.64 %", "3-seed ensemble; 95 % interval 99.02 to 99.99")
        + stat("43", "errors in 12,000 images, all from 4 of the 420 drawings")
        + stat("99.9 %", "on the original Kaggle split (reference notebook: 96.5 %)") + '</div>'
        + '<div style="display:flex;gap:32px">'
        + '<div style="display:flex;flex-direction:column;gap:12px">' + img("writers", 800, 340, "Bar chart of accuracy on held-out writers: AksaraNet 100, 99.5, 100, 98.7 and 98.8 percent; reference Kaggle CNN 62.2, 67.8, 80.8, 94.8 and 43.2 percent")
        + p("<b>Unseen writers:</b> 98.7 to 100 %, where the reference CNN drops to 43 to 95 %.", 26, INK, "width:800px") + '</div>'
        + '<div style="flex:1;display:flex;flex-direction:column;gap:12px">' + img("words", 832, 340, "Eight photographed handwritten words with detected glyph boxes and predicted labels; seven words are read correctly and jayabaya is read as jayanyaya")
        + p("<b>Word photos:</b> 7 of 8 words and 27 of 28 glyphs read correctly (reference segmentation: 5 of 8).", 26, INK) + '</div></div>')
slide("results3", 9, "Experiment 3: the final model generalises", body,
      "Results three. The final model is tested on every drawing exactly once. Errors concentrate in four unusual drawings. On the original split our re-implementation of the reference notebook gives 96.8, matching its published 96.5, and our model gives 99.9. Holding out each writer family shows the gain is robustness, not memorisation. For words, a dynamic programme lets the classifier decide which strokes form one glyph, fixing the two-stroke characters the reference could not segment.", "Results 3 of 3")

# 10 conclusion -----------------------------------------------------------
concl = [("Audit the data first", "12,000 images were 420 drawings. Splitting by drawing changed an MLP's score from 95 % to 53 %."),
         ("Architecture over tuning", "BatchNorm and global average pooling took the CNN from 87 % to 98.5 % with 7× fewer weights."),
         ("Tuning adds the last point", "AdamW, one-cycle schedule and strong augmentation: 98.5 % to 99.4 %, and 98.7 to 100 % on unseen writers.")]
body = ('<div style="display:flex;gap:24px">' + "".join(card(t, p(b, 28), accent=ACC2) for t, b in concl) + '</div>'
        + '<div style="display:flex;gap:24px;flex:1">'
        + card("Limitations", ul(["Only 420 drawings by about five writers", "Accuracy is at the dataset's ceiling, so good recipes cannot be ranked", "Base characters only: no sandhangan, pasangan or numerals", "Word reading tested on eight clean photos"], 26))
        + card("Future work", ul(["Collect handwriting from more writers", "Extend the label set to vowel marks and conjuncts", "Sequence model (CNN + recurrent layers, CTC) on whole words", "Compare with transfer learning from a pretrained backbone"], 26))
        + '</div>')
slide("conclusion", 10, "Conclusion", body,
      "Three messages: check where the data comes from, architecture mattered more than hyper-parameters, and augmentation is what makes the model robust to new writers. Then limitations and next steps.", "Conclusion")

# 11 code -----------------------------------------------------------------
IMG["keras"] = "aa46cdb41c03fb33075c62e3889f73b8"
body = (f'<div style="display:flex;gap:32px;align-items:center;background:{CARD};padding:22px 36px;border:1px solid {LINE};border-left:8px solid {ACC};border-radius:12px">'
        f'<p style="font-size:24px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:{MUTED}">Repository</p>'
        f'<p style="{HEAD};font-size:44px;font-weight:600;color:{ACC}">[GitHub URL of the group repository]</p></div>'
        + img("keras", 1664, 420, "Grouped bar charts of test accuracy per fold for PyTorch and Keras. Baseline training: means 98.5 and 98.0 percent. Tuned training: means 99.5 and 99.0 percent. The largest gap is on fold 2 of tuned training, 98.8 versus 97.0")
        + p("<b>Reproduced in a second framework.</b> The same network, training and folds in Keras 3 give 99.0 % against 99.5 % in PyTorch with tuned training: about half a point, inside seed-to-seed noise (one seed each).", 26)
        + p("Repository: executed notebook, <b>hanacaraka/</b> (PyTorch), <b>hanacaraka_keras/</b> (Keras), experiment scripts and the log of every training run.", 24, MUTED))
slide("code", 11, "Code and reproducibility", body, "Everything on these slides is produced by the notebook from the run log. As a reproducibility check we ported the model to Keras, the framework used in the lectures, and trained it on the same five folds. Both frameworks agree within noise; the one visible gap, fold 2 of tuned training, is a fold where PyTorch's own seeds range from 97.2 to 99.3. Keras ran on CPU, so timings are not comparable. Replace the placeholder with the repository link before presenting.", "URL of the code")

# 12 references -----------------------------------------------------------
refs = ["Javanese Script (Aksara Jawa) Augmented, Kaggle dataset, and its reference notebook (CNN, 96.5 % test accuracy)",
        "Afiahayati, Introduction to Deep Learning, lecture slides, Universitas Gadjah Mada",
        "LeCun et al., Gradient-based learning applied to document recognition, Proc. IEEE, 1998",
        "Simonyan and Zisserman, Very deep convolutional networks for large-scale image recognition, ICLR 2015",
        "He et al., Deep residual learning for image recognition, CVPR 2016",
        "Ioffe and Szegedy, Batch normalization, ICML 2015",
        "Lin et al., Network in network (global average pooling), ICLR 2014",
        "Kingma and Ba, Adam, ICLR 2015; Loshchilov and Hutter, Decoupled weight decay regularization, ICLR 2019",
        "Smith and Topin, Super-convergence (one-cycle schedule), 2018",
        "Dalal and Triggs, Histograms of oriented gradients for human detection, CVPR 2005",
        "Kaufman et al., Leakage in data mining, ACM TKDD, 2012",
        "Simard et al., Best practices for convolutional neural networks applied to visual document analysis, ICDAR 2003"]
body = f'<ol style="font-size:25px;line-height:1.5;color:{INK}">' + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"
slide("references", 12, "References", body, "References for the dataset, the course material and every method used.", "References")

for sid, html in slides:
    open(os.path.join(D, "project", "slides", f"{sid}.html"), "w").write(html)
deck = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-10-11T13:30:00Z"}, "lists": "css",
        "title": "Aksara Jawa Recognition - Group Mini Project", "order": [s for s, _ in slides], "cover": "cover",
        "sections": {"s1": {"description": "Who we are and what problem we solve.", "start": "cover"},
                     "s2": {"description": "The data, the split, the model and how it is trained.", "start": "data"},
                     "s3": {"description": "Three experiments: architecture, tuning, final model.", "start": "results1"},
                     "s4": {"description": "Conclusion, code and references.", "start": "conclusion"}},
        "faces": {"source-serif-4": {"family": "Source Serif 4", "href": "https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
                  "inter": {"family": "Inter", "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400..700&display=swap"}},
        "designSystems": []}
json.dump(deck, open(os.path.join(D, "project", "deck.json"), "w"), indent=1)
print(len(slides), [s for s, _ in slides])
