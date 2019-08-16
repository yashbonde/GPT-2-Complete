"""
clean_nips.py

Why? Because bored in office.

@yashbonde - 14.09.2019
"""

import logging
import json
import sentencepiece as spm
import argparse
import os
from tqdm import tqdm
import regex as re

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description = 'Script to clean the downloaded NIPS papers data. The behaviour is slightly '
                                                  'unpredictable and depends upon your vocabulary.')
    parse.add_argument('--folder', default = './working', help = 'Folder with data, we will "walk" over this folder')
    parse.add_argument('--model', default='nips', help='Name of sentencepiece model')
    parse.add_argument('--sentence_length', default = 5000, type = int, help = 'Number of characters to keep in each sequence')
    parse.add_argument('--hard_vocab_limit', default = False, type = bool,
                       help = 'If the text cannot be split into a given vocab size the sentence piece returns an error,'
                              ' this is used to fix that problem. But this can result in dynamic sized vocabulary.')
    parse.add_argument('--vocab_size', default= int(8129 * 0.2), type = int, help = 'Size of the vocabulary to use')
    args = parse.parse_args()

    # find all jsons which have data
    all_json_paths = []
    for root_dir, _, path in os.walk(args.folder):
        for p in path:
            if p.split('.')[-1] == 'json':
                all_json_paths.append(os.path.join(root_dir, p))

    # iterate over all files and combine `abstract` along with `text`
    '''
    curr_paper_data = {
            'link': pdf_link,
            'pdf_path': pdf_path,
            'paper_id': paper_id,
            'year': year,
            'paper_title': paper_title,
            'event_type': event_type,
            'pdf_name': pdf_name,
            'abstract': abstract,
            'paper_text': ftfy.fix_text(paper_text.decode('utf-8')),
            'authors': authors
        }
    '''
    re_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    all_txt = []
    for i in tqdm(range(len(all_json_paths))):
        try:
            curr_json = json.load(open(all_json_paths[i]))
        except:
            print('* Failed on file:', all_json_paths[i])
            pass
        t_ = curr_json['abstract'] + ' ' + curr_json['paper_text']
        t_ = ''.join(re.findall(re_pat, t_))
        t_ = t_.replace('\n', ' ')[:args.sentence_length]
        all_txt.append(t_)

    # combine all to single text block
    all_tt = '\n'.join(all_txt)
    name_file = os.path.join(os.getcwd(), 'nips_clean.txt')
    logging.warning('Writing output file to {}'.format('nips_clean.txt'))
    with open(name_file, 'w', encoding='utf-8') as f:
        f.write(all_tt)

    # make sentencepiece model
    hv = 'false' if args.hard_vocab_limit else 'true'
    spm.SentencePieceTrainer.train('--input={} \
                                    --model_prefix={} \
                                    --vocab_size={} \
                                    --normalization_rule_name=nfkc_cf\
                                    --pad_id={}\
                                    --hard_vocab_limit={}'.format(name_file,
                                                                     args.model,
                                                                     args.vocab_stze,
                                                                     3,
                                                                     hv
                                                                  ))

