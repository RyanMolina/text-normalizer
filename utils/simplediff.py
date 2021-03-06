from collections import namedtuple

TestStatistics = namedtuple('TestStatistics', ['fp', 'tn', 'tp', 'fn'])


def lcs(a, b):
    if not a or not b:
        return []
    l = len(a) + len(b) - 1
    # Fill non-comparable elements with null spaces.
    sa = a + (len(b) - 1) * ['']
    sb = (len(a) - 1) * [''] + b
    longest = []
    for k in range(l):
        cur = []
        for c in range(l):
            if sa[c] != '' and sb[c] != '' and sa[c] == sb[c]:
                cur.append(sa[c])
            else:
                if len(cur) > len(longest):
                    longest = cur
                cur = []
        if len(cur) > len(longest):
            longest = cur

        if sa[len(sa) - 1] == '':
            # Shift 'a' to the right.
            sa = [''] + sa[: len(sa) - 1]
        else:
            # Shift 'b' to the left.
            sb = sb[1:] + ['']
    return longest


def find_sub_list(l, sub):
    if len(sub) > len(l):
        return -1
    for i in range(len(l) - len(sub) + 1):
        j = 0
        eq = True
        for s in sub:
            if l[i + j] != s:
                eq = False
                break
            j += 1
        if eq:
            return i
    return -1


def align_chars(sequence1, sequence2):
    # lcs is the Longest Common Subsequence function.
    cs = lcs(sequence1, sequence2)

    if not cs:
        return sequence1 + [''] * len(sequence2), \
               [''] * len(sequence1) + sequence2
    else:
        # Remainings non-aligned sequences in the left side.
        left1 = sequence1[:find_sub_list(sequence1, cs)]
        left2 = sequence2[:find_sub_list(sequence2, cs)]

        # Remainings non-aligned sequences in the right side.
        right1 = sequence1[find_sub_list(sequence1, cs) + len(cs):]
        right2 = sequence2[find_sub_list(sequence2, cs) + len(cs):]

        # Align the sequences in the left and right sides:
        left_align = align_chars(left1, left2)
        right_align = align_chars(right1, right2)

        return left_align[0] + cs + right_align[0], \
               left_align[1] + cs + right_align[1]


def align_tokens(a, b):
    for i in range(len(a)):
        if b[i] == '-' and a[i] == ' ':
            if '-' == b[i - 1]:
                a[i] = '_'
            else:
                b[i] = ' '
        elif b[i] == ' ' and a[i] == '-':
            if '-' == a[i - 1]:
                b[i] = '_'
            else:
                a[i] = ' '


def check_errors(enc, dec, res, tagged_words):
    enc = enc.replace('-', '<hyphen>')
    res = res.replace('-', '<hyphen>')
    dec = dec.replace('-', '<hyphen>')

    tagged_enc = []
    tags_count = {
        'accent_styles': {'fp': 0, 'tp': 0, 'tn': 0},
        'phonetic_styles': {'fp': 0, 'tp': 0, 'tn': 0},
        'contractions': {'fp': 0, 'tp': 0, 'tn': 0},
        'repeating_characters': {'fp': 0, 'tp': 0, 'tn': 0},
        'repeating_units': {'fp': 0, 'tp': 0, 'tn': 0},
        'misspellings': {'fp': 0, 'tp': 0, 'tn': 0},
    }

    a, b = align_chars(list(res), list(dec))
    res_dec = ['-' if i == '' else i for i in a]
    dec_res = ['-' if j == '' else j for j in b]

    align_tokens(res_dec, dec_res)
    dec_res = ''.join(dec_res).split()
    res_dec = ''.join(res_dec).split()

    a, b = align_chars(list(enc), list(dec))
    enc_dec = ['-' if i == '' else i for i in a]
    dec_enc = ['-' if j == '' else j for j in b]

    align_tokens(enc_dec, dec_enc)
    dec_enc = ''.join(dec_enc).split()
    enc_dec = ''.join(enc_dec).split()

    tp = 0
    tn = 0
    fp = 0

    output = []
    for i in range(len(res_dec)):
        try:
            res_dec[i] = res_dec[i].replace('_', ' ').replace('-', '').replace('<hyphen>', '-')
            dec_res[i] = dec_res[i].replace('_', ' ').replace('-', '').replace('<hyphen>', '-')
            dec_enc[i] = dec_enc[i].replace('_', ' ').replace('-', '').replace('<hyphen>', '-')
            enc_dec[i] = enc_dec[i].replace('_', ' ').replace('-', '').replace('<hyphen>', '-')
        except IndexError:
            pass

        try:
            tag = tagged_words.get(enc_dec[i].lower())
            if tag:
                tagged_enc.append('<abbr title="{}">{}</abbr>'.format(tag, enc_dec[i]))
            else:
                tagged_enc.append(enc_dec[i])

            if not res_dec[i] and enc_dec[i] == dec_enc[i]:
                """Checks if the result is not in the expected output."""
                if tag:
                    tags_count[tag]['fp'] = tags_count.get(tag).get('fp', 0) + 1

                if res_dec[i]:
                    output.append('<span class="false-pos">{}</span>'.format(res_dec[i]))
                fp += 1
            elif enc_dec[i] == dec_enc[i] and res_dec[i] != dec_res[i]:
                """Checks if the already normalized input have been informalized"""
                if tag:
                    tags_count[tag]['fp'] = tags_count.get(tag).get('fp', 0) + 1

                output.append('<span class="false-pos">{}</span>'.format(res_dec[i]))
                fp += 1
            elif dec_res[i] == res_dec[i]:
                """Checks if the result is equal to expected output."""
                if tag:
                    tags_count[tag]['tp'] = tags_count.get(tag).get('tp', 0) + 1

                output.append(res_dec[i])
                tp += 1
            else:
                """The result is not the same with the expected output."""
                output.append('<span class="true-neg">{}</span>'.format(res_dec[i]))
                tn += 1
        except IndexError:
            """Marks extra words as errors."""
            if res_dec[i] != dec_res[i]:
                output.append('<span class="true-neg">{}</span>'.format(res_dec[i]))
            else:
                output.append(res_dec[i])
            tn += 1
    return ' '.join(output), TestStatistics(tp=tp, fn=0, tn=tn, fp=fp), ' '.join(tagged_enc), tags_count
