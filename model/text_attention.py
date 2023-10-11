import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, true_label, labels, latex_file, color='red', rescale_value = False):
    print(attention_list.shape, len(labels), len(text_list))
    assert(len(text_list) == len(attention_list[0]), f"len(text_list)={len(text_list)} != len(attention_list[0])={len(attention_list[0])}")
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        
        for i, label in enumerate(labels):
            if label not in true_label:
                continue
            string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
            for idx in range(word_num):
                string += "\\colorbox{%s!%s}{"%(color, float(attention_list[i][idx].item()) * 10000)+"\\strut " + text_list[idx]+"} "
            string += "\n}}}"
            f.write(string+'\n')
            if label in true_label:
                # write bold label
                f.write(r'''\textbf{%s}'''%label)
            else:
                f.write(r'''%s'''%label)

        f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list