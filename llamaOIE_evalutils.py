import sys
from nltk.stem import WordNetLemmatizer
from Levenshtein import distance as levenstein_distance
from typing import List
from itertools import permutations
import copy
import signal

def handler(signum, frame):
    raise TimeoutError("Code timed out!")

def tokenize_triple_perms(tpl_dict: dict, tokenizer, perm: bool) -> List[str]:
    perm_candidates = []
    if perm:
        obj_perms = list(permutations(tpl_dict['objs']))
    else:
        obj_perms = [tpl_dict['objs']]
    for objperm in obj_perms:
        tpl_sent = f"{tpl_dict['subj']} {' '.join(tpl_dict['auxi'])+' ' if len(tpl_dict['auxi']) > 0 else ''}{tpl_dict['pred']}" \
                f" {' '.join([' '.join(obj) for obj in objperm])}"
        tpl_sent = tpl_sent.strip(' ')
        tpl_sent = tokenizer.tokenize(tpl_sent)
        perm_candidates.append(tpl_sent)
    if perm:
        return perm_candidates
    else:
        return perm_candidates[0]


def parse_outstr_to_triples(oie_str: str):
    """Parse the output string from the model to a list of triples.
    """
    oie_str = oie_str.lower()
    output_triples = []

    tpl_strs = oie_str.split('\n')
    nlst = []
    for tstr in tpl_strs:
        if tstr.strip(' ') == '':
            continue
        elif tstr in nlst:
            continue
        else:
            nlst.append(tstr)
    tpl_strs = copy.deepcopy(nlst)
    del nlst

    for tidx, tstr in enumerate(tpl_strs):
        if tidx > 0:
            t_list = tstr.split('.')
            if not t_list[0].isdigit():
                # print(f"Warning: {tstr}: not a valid triple string", file=sys.stderr)
                continue
            tstr = '.'.join(t_list[1:])
            del t_list
        else:
            pass
        tpl_argslist = tstr.split(',,')
        if len(tpl_argslist) < 2:
            # print(f"Warning: {tstr}: not a valid triple string", file=sys.stderr)
            continue
        subj = tpl_argslist[0].strip(' ')
        pred = tpl_argslist[1].strip(' ')
        pred_lst = pred.split('###')
        pred_lst = [x.strip(' ') for x in pred_lst]
        auxi = pred_lst[:-1]
        pred = pred_lst[-1]

        objs_lst = tpl_argslist[2:]
        objs = []
        for obj in objs_lst:
            curr_obj_tup = obj.split('###')
            if len(curr_obj_tup) == 1:
                prep = ""
                obj = curr_obj_tup[0]
            elif len(curr_obj_tup) >= 2:
                if len(curr_obj_tup) > 2:
                    # print(f"Warning: {curr_obj_tup} length larger than 2! Only the last two are kept.", file=sys.stderr)
                    pass
                prep = curr_obj_tup[-2]
                obj = curr_obj_tup[-1]
            else:
                # print(f"Warning: {tstr}: not a valid triple string", file=sys.stderr)
                continue
            prep = prep.strip(' ')
            obj = obj.strip(' ')
            objs.append((prep, obj))
        output_triples.append({'subj': subj, 'pred': pred, 'auxi': auxi, 'objs': objs})
    return output_triples


# Borrowed from CaRB matcher.py
def tuple_match(ref, ex, lemmatizer, element_weights=None):
    def single_match(wref: List[str], wex: List[str], curr_w: float):
        """Match two lists of words, and return the number of matching words.
        """
        precision[1] += len(wex) * curr_w
        recall[1] += len(wref) * curr_w
        wref = [lemmatizer.lemmatize(w) for w in wref]
        wex = [lemmatizer.lemmatize(w) for w in wex]
        matching_words = 0
        for w in wref:
            if w in wex:
                matching_words += curr_w
                wex.remove(w)
        precision[0] += matching_words
        recall[0] += matching_words
        return matching_words == 0

    precision = [0, 0] # 0 out of 0 predicted words match
    recall = [0, 0] # 0 out of 0 reference words match
    # If, for each part, any word is the same as a reference word, then it's a match.
    if element_weights is None:
        element_weights = {'pred': 1.0, 'subj': 1.0, 'obj': 1.0, 'auxi': 1.0}
    else:
        assert isinstance(element_weights, dict)
        assert set(element_weights.keys()) == {'pred', 'subj', 'obj', 'auxi'}

    for ele in ['subj', 'pred']:
        predicted_words = ex[ele].split()
        gold_words = ref[ele].split()
        failure_flag = single_match(gold_words, predicted_words, element_weights[ele])
        if failure_flag:
            return None
    
    for ref_aux, ex_aux in zip(ref['auxi'], ex['auxi']):
        predicted_words = ex_aux.split()
        gold_words = ref_aux.split()
        failure_flag = single_match(gold_words, predicted_words, element_weights['auxi'])
        if failure_flag:
            pass  # we don't return None here because auxiliaries are not required to match (?)
            # return None
    
    if len(ref['objs']) != len(ex['objs']):
        if len(ex['objs']) == 0:
            return None
        else:
            ref['objs'] = ref['objs'][:len(ex['objs'])]
    
    for ref_obj, ex_obj in zip(ref['objs'], ex['objs']):
        assert isinstance(ref_obj, tuple) and len(ref_obj) == 2
        assert isinstance(ex_obj, tuple) and len(ex_obj) == 2
        ref_obj = ' '.join(ref_obj)
        ex_obj = ' '.join(ex_obj)
        predicted_words = ex_obj.split()
        gold_words = ref_obj.split()
        failure_flag = single_match(gold_words, predicted_words, element_weights['obj'])
        if failure_flag:
            return None

    prec = 1.0 * precision[0] / precision[1]
    rec = 1.0 * recall[0] / recall[1]
    return [prec, rec]


def get_match_sum(len_gold: int, len_pred: int, vals_matrix: List[list],
                  higher_is_better=True):
    score_sum = 0
    matched_scores = []
    # each index in the following pair of lists corresponds to a match in the plan.
    selected_rows = []
    selected_cols = []
    num_precision_matches = min(len_gold, len_pred)

    for m in range(num_precision_matches):
        matched_row = -1
        matched_col = -1
        matched_scr = None
        for i in range(len_gold):
            if i in selected_rows:
                continue
            for j in range(len_pred):
                if j in selected_cols or vals_matrix[i][j] is None:
                    continue
                if matched_scr is None or (vals_matrix[i][j] > matched_scr and higher_is_better) or \
                        (vals_matrix[i][j] < matched_scr and not higher_is_better):
                    matched_scr = vals_matrix[i][j]
                    matched_row = i
                    matched_col = j
        if matched_scr is None:
            print("No match found before all matches are exhausted.", file=sys.stderr)
        else:
            selected_rows.append(matched_row)
            selected_cols.append(matched_col)
            score_sum += matched_scr
            matched_scores.append(matched_scr)
    return score_sum, matched_scores


def compare_prediction_gold(pred_str: str, gold_str: str, lemmatizer: WordNetLemmatizer, 
                            element_weights=None, tokenizer=None, f_score_beta=1.0):
    # lemmatizer = WordNetLemmatizer()
    # element_weights = {'pred': 1.0, 'subj': 1.0, 'obj': 1.0, 'auxi': 1.0}
    pred_triples = parse_outstr_to_triples(pred_str)
    gold_triples = parse_outstr_to_triples(gold_str)
    scores_matrix = [[None for _ in pred_triples] for __ in gold_triples]

    for gidx, gold_tpl in enumerate(gold_triples):
        for pidx, pred_tpl in enumerate(pred_triples):
            curr_score = tuple_match(gold_tpl, pred_tpl, lemmatizer, element_weights)
            scores_matrix[gidx][pidx] = curr_score

    recall_numerator = 0
    for i, row in enumerate(scores_matrix):
        recall_rowmax = max([x[1] for x in row if x is not None], default=0)
        recall_numerator += recall_rowmax
    
    precs_matrix = [[y[0] if y is not None else None for y in x] for x in scores_matrix]
    precision_numerator, _ = get_match_sum(len(gold_triples), len(pred_triples), precs_matrix, higher_is_better=True)

    precision_denominator = len(pred_triples)
    recall_denominator = len(gold_triples)
    prec = 1.0 * precision_numerator / precision_denominator if precision_denominator > 0 else 1.0
    rec = 1.0 * recall_numerator / recall_denominator if recall_denominator > 0 else 0.0
    f_score = (1 + f_score_beta ** 2) * prec * rec / (f_score_beta ** 2 * prec + rec) if prec + rec > 0 else 0.0

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(20)
    try:
        levenshtein_matrix = [[None for _ in pred_triples] for __ in gold_triples]
        for gidx, gold_tpl in enumerate(gold_triples):
            for pidx, pred_tpl in enumerate(pred_triples):
                # Calculate the minimum Levenshtein distance between the gold and predicted triples (with any permutation of objects)
                min_scr = None
                gold = tokenize_triple_perms(gold_tpl, tokenizer, perm=False)
                pred_perms = tokenize_triple_perms(pred_tpl, tokenizer, perm=True)
                for pperm in pred_perms:
                    curr_score = levenstein_distance(gold, pperm)
                    if min_scr is None or curr_score < min_scr:
                        min_scr = curr_score
                levenshtein_matrix[gidx][pidx] = min_scr
        _, levenshtein_dists = get_match_sum(len(gold_triples), len(pred_triples), levenshtein_matrix, higher_is_better=False)
        # TODO: select the best match for each gold triple
    except TimeoutError as te:
        print(te)
        min_len = min(len(gold_triples), len(pred_triples))
        levenshtein_dists = [20 for x in range(min_len)]
    finally:
        # Disable the alarm
        signal.alarm(0)

    return {'prec_num': precision_numerator, 'prec_den': precision_denominator, 'prec': prec,
            'rec_num': recall_numerator, 'rec_den': recall_denominator, 'rec': rec,
            'f_score': f_score, 'f_beta': f_score_beta, 'levenshtein_dists': levenshtein_dists
            }


if __name__ == '__main__':
    # For debugging purposes
    a = parse_outstr_to_triples("""A casting director ,,  is ,, at ### the time
2. the casting director ,,  told ,, Scott ,, that ### he had wished that he 'd met him a week before
3. Scott ,,  met ,, the casting director ,, a week before
4. the casting director ,,  is casting ,, for ### the `` G.I. Joe '' cartoon
5. the `` G.I. Joe '' cartoon ,,  is casting ,, for ### the `` G.I. Joe '' cartoon""")
    print(a)