from tqdm import tqdm
import ujson as json


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = []
        if self.args.input_format == 'entity_marker':
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        self.tokenizer.add_tokens(self.new_tokens)
        if self.args.input_format not in ('typed_entity_marker_punct',):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe):
        """
        Implement the following input formats:
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        sents = []

        input_format = self.args.input_format

        subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
        obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'typed_entity_marker_punct':
                if i_t == ss:
                    temp = ['@'] + ['*'] + subj_type + ['*']
                    new_ss_mask = len(sents) + len(temp) - 1
                    tokens_wordpiece = temp + tokens_wordpiece
                if i_t == se:
                    new_se_mask = len(sents) + len(tokens_wordpiece)
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    temp = ["#"] + ['^'] + obj_type + ['^']
                    new_os_mask = len(sents) + len(temp) - 1
                    tokens_wordpiece = temp + tokens_wordpiece
                if i_t == oe:
                    new_oe_mask = len(sents) + len(tokens_wordpiece)
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)

        sents = sents[:self.args.max_seq_length - 2]

        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        s_mask = [0] * len(input_ids)
        o_mask = [0] * len(input_ids)

        new_se_mask = new_se_mask + 1
        new_oe_mask = new_oe_mask + 1
        new_ss_mask = new_ss_mask + 1
        new_os_mask = new_os_mask + 1
        for i in range(new_ss_mask + 1, new_se_mask):
            s_mask[i] = 1
        for i in range(new_os_mask + 1, new_oe_mask):
            o_mask[i] = 1

        return input_ids, s_mask, o_mask


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.label2id = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(self.label2id)

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, s_mask, o_mask = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.label2id[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                's_mask': s_mask,
                'o_mask': o_mask,
                'guid': d['id']
            }

            features.append(feature)
        return features


class RETACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.label2id = {'no_relation': 0, 'org:founded_by': 1, 'per:identity': 2, 'org:alternate_names': 3, 'per:children': 4, 'per:origin': 5, 'per:countries_of_residence': 6, 'per:employee_of': 7, 'per:title': 8, 'org:city_of_branch': 9, 'per:religion': 10, 'per:age': 11, 'per:date_of_death': 12, 'org:website': 13, 'per:stateorprovinces_of_residence': 14, 'org:top_members/employees': 15, 'org:number_of_employees/members': 16, 'org:members': 17, 'org:country_of_branch': 18, 'per:spouse': 19, 'org:stateorprovince_of_branch': 20, 'org:political/religious_affiliation': 21, 'org:member_of': 22, 'per:siblings': 23, 'per:stateorprovince_of_birth': 24, 'org:dissolved': 25, 'per:other_family': 26, 'org:shareholders': 27, 'per:parents': 28, 'per:charges': 29, 'per:schools_attended': 30, 'per:cause_of_death': 31, 'per:city_of_death': 32, 'per:stateorprovince_of_death': 33, 'org:founded': 34, 'per:country_of_death': 35, 'per:country_of_birth': 36, 'per:date_of_birth': 37, 'per:cities_of_residence': 38, 'per:city_of_birth': 39}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(self.label2id)

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            input_ids, s_mask, o_mask = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)
            rel = self.label2id[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                's_mask': s_mask,
                'o_mask': o_mask,
                'guid': d['id']
            }

            features.append(feature)
        return features
