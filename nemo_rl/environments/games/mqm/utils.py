import json
import random
from collections import defaultdict
import re

lang_lookup = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "cz": "Czech",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "is": "Icelandic",
    "fi": "Finnish",
    "pl": "Polish",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "et": "Estonian",
    "ta": "Tamil",
    "gu": "Gujarati",
    "hi": "Hindi",
    "bn": "Bengali",
    "kk": "Kazakh",
    "uk": "Ukrainian",
    "tr": "Turkish",
    "ha": "Hausa",
    "ps": "Pashto",
    "km": "Khmer",
    "zu": "Zulu",
    "xh": "Xhosa"
}

# TODO: support char langs
def span_to_pos(text):
    start_tag = "<v>"
    end_tag = "</v>"

    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1:
        return []

    # Extract prefix, span text, and suffix
    before = text[:start_idx]
    span_text = text[start_idx + len(start_tag):end_idx]

    # Tokenize everything into words
    words = text.replace(start_tag, "").replace(end_tag, "").split()
    before_words = before.split()
    span_words = span_text.split()

    # Word positions of the span
    start_pos = len(before_words)
    span_positions = list(range(start_pos, start_pos + len(span_words)))
    return span_positions

def extract_span(error):
    start_tag = "<v>"
    end_tag = "</v>"

    text = error["src_span"]
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        pass
    else:
        return "source", text[start_idx + len(start_tag):end_idx]

    text = error["tgt_span"]
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx == -1 or end_idx == -1:
        return None, None
    else:
        return "target", text[start_idx + len(start_tag):end_idx]  

class MQMExampleGenerator:
    def __init__(self, filepath, n=3, span_type="index", _filter=0, think_egs=0, use_ref=0, specialize=0):
        """
        Initialize with the path to the JSONL file.
        """
        self.filepath = filepath
        self.n = n
        self.span_type = span_type
        self.use_ref = use_ref
        self.data = self._load_data(seg_by_seg=(specialize==3))
        self.specialize = specialize
        if self.specialize != 0:
            self.data_dict = self._build_data_dict()
            self.backup_data = self._load_data_backup()
        if _filter > 0:
            self._filter_data(_filter)
        if think_egs == 1:
            self._build_think_egs()
    
    def _build_error_id_egs(self, error_types_in_line, error_type_to_find, errors):
        rv = []
        if error_type_to_find in error_types_in_line:
            for error in errors:
                if error["category"] == error_type_to_find:
                    src_or_tgt, span = extract_span(error)
                    if span == None:
                        continue
                    rv.append(f"There is a {error['severity']} {error['category']} error with the {src_or_tgt} span: <v> {span} </v>.")
        if len(rv) == 0:
            return f" There are no {error_type_to_find} errors."
        else:
            return " "+" ".join(rv)

    def _build_error_id_egs_other(self, errors, error_types_covered=["accuracy", "fluency", "terminology", "locale convention"]):
        rv = []
        for error in errors:
            if error["category"] not in error_types_covered and (error["category"] != "no-error" or error["category"] != "no error"):
                src_or_tgt, span = extract_span(error)
                rv.append(f"There is a {error['severity']} {error['category']} error with the {src_or_tgt} span: <v> {span} </v>.")
        if len(rv) == 0:
            return " There are no other errors."
        else:
            return " "+" ".join(rv)

    def _build_think_egs(self):
        for idx, (input_text, output_text, line_hash, line) in enumerate(self.data):
            # header
            # src_text = line[""]
            header = f"Okay, let's take a look at the translation provided. The English source is: {line['src']} The German translation is: {line['tgt']}."
            error_types_in_line = set([x["category"] for x in line["errors"]])

            # build error identification
            accuracy = "First, I'll check for accuracy errors." + self._build_error_id_egs(error_types_in_line, "accuracy", line["errors"])
            fluency = "Second, I'll check for fluency errors." + self._build_error_id_egs(error_types_in_line, "fluency", line["errors"])
            terminology = "Third, I'll check for terminology errors." + self._build_error_id_egs(error_types_in_line, "terminology", line["errors"])
            locale = "Fourth, I'll check for locale convention errors." + self._build_error_id_egs(error_types_in_line, "locale convention", line["errors"])
            other = "Finally, I'll check for other errors." + self._build_error_id_egs_other(line["errors"])
            
            think = "<t>"+" ".join([header, accuracy, fluency, terminology, locale, other])+"</t>"
            # TODO: build output construction

            self.data[idx][1] = think + " The final answer is:\n\\boxed{" + output_text + "}"

    def _filter_data(self, n):
        self.data.sort(key=len)
        self.data = self.data[-n:]

    def _load_data(self, seg_by_seg=False):
        """
        Load and parse the JSONL file.
        """
        data = []
        no_err_lines = 0
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = json.loads(line)
                    if line["errors"] is None:
                        no_err_lines += 1
                        continue
                    if not seg_by_seg:
                        input_text = self._format_input(line)
                        output_text = self._format_output(line)
                        line_hash = self._line_hash(line)
                        data.append([input_text, output_text, line_hash, line])
                    else:
                        # import pdb;pdb.set_trace()
                        line_hash = self._line_hash(line)
                        input_texts = []
                        output_texts = []
                        assert len(line["src_seg"]) == len(line["tgt_seg"])
                        assert len(line["src_seg"]) == len(line["errors_seg"])
                        for (src, tgt, errors) in zip(line["src_seg"], line["tgt_seg"], line["errors_seg"]):
                            seg_line = {"src": src, "tgt": tgt, "errors": errors, "lp": line["lp"]}
                            input_text = self._format_input(seg_line)
                            output_text = self._format_output(seg_line)
                            input_texts.append(input_text)
                            output_texts.append(output_text)
                        data.append([input_texts, output_texts, line_hash, line])
        print(f"Skipped {no_err_lines} egs line with no error annotation")
        return data
    
    def _load_data_backup(self):
        """
        Load and parse the JSONL file.
        """
        data = []
        no_err_lines = 0
        with open("data/gemba_examples.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    line = json.loads(line)
                    if line["errors"] is None:
                        no_err_lines += 1
                        continue
                    input_text = self._format_input(line)
                    output_text = self._format_output(line)
                    line_hash = self._line_hash(line)
                    data.append([input_text, output_text, line_hash, line])
        print(f"Skipped {no_err_lines} egs line with no error annotation")
        return data

    def _line_hash(self, example):
        """hash is based on src text"""
        return example["src"]

    def _format_input(self, example):
        """
        Create a textualized input string from one JSON example.
        Modify this function based on your schema.
        """
        src_lang = lang_lookup[example["lp"].split("-")[0]]
        tgt_lang = lang_lookup[example["lp"].split("-")[-1]]
        src_text = example["src"]
        tgt_text = example["tgt"]
        if self.use_ref == 1:
            ref_text = example["ref"]
            return f"{src_lang} source:\n'''{src_text}'''\n{tgt_lang} translation:\n'''{tgt_text}'''\n{tgt_lang} reference:\n'''{ref_text}'''\n"
        else:
            return f"{src_lang} source:\n'''{src_text}'''\n{tgt_lang} translation:\n'''{tgt_text}'''\n"

    def _format_output(self, example):
        """
        Create a textualized output string from one JSON example.
        Modify this to reflect how you want to represent error annotations.
        """
        rv = {"errors": []}
        for error in example["errors"]:
            output = {"severity": error["severity"], "category": error["category"]}
            if self.span_type == "none":
                rv["errors"].append(output)
                continue

            if "<v>" in error["src_span"] and "<v>" in error["tgt_span"]:
                # not expecting two error spans
                import pdb;pdb.set_trace()
            elif "<v>" in error["src_span"]:
                if self.span_type == "index":
                    output["source_or_target"] = "source"
                    pos = span_to_pos(error["src_span"])
                    if len(pos) == 0:
                        continue
                    output["text"] = " ".join(example["src"].split()[pos[0]:pos[-1]+1])
                    output["start"] = pos[0]
                    output["end"] = pos[-1]
                elif self.span_type == "tag":
                    output["src_span"] = " ".join(error["src_span"].replace("<v>"," <v> ").replace("</v>"," </v> ").split())
                elif self.span_type == "seg":
                    matches = re.findall(r"<v>(.*?)</v>", error["src_span"])
                    if matches:
                        output["src_span"] = matches[0]
                else:
                    print(f"invalid span type: {self.span_type}")
            elif "<v>" in error["tgt_span"]:
                if self.span_type == "index":
                    output["source_or_target"] = "target"
                    pos = span_to_pos(error["tgt_span"])
                    if len(pos) == 0:
                        continue
                    output["text"] = " ".join(example["tgt"].split()[pos[0]:pos[-1]+1])
                    output["start"] = pos[0]
                    output["end"] = pos[-1]
                elif self.span_type == "tag":
                    output["tgt_span"] = " ".join(error["tgt_span"].replace("<v>"," <v> ").replace("</v>"," </v> ").split())
                elif self.span_type == "seg":
                    matches = re.findall(r"<v>(.*?)</v>", error["tgt_span"])
                    if matches:
                        output["tgt_span"] = matches[0]
                else:
                    print(f"invalid span type: {self.span_type}")
            else:
                # no error span
                continue
            rv["errors"].append(output)

        return json.dumps(rv, ensure_ascii=False)

    def _build_data_dict(self):
        data_dict = defaultdict(list)
        for d in self.data:
            data_dict[d[2]].append(d)
        return data_dict

    def get_examples(self, test_egs):
        """
        Returns a list of (input, output) tuples.
        """
        if self.specialize == 1:
            # this is a dummy test; return all ic examples except the ones which match the tgt
            ic_egs = self.data_dict[test_egs[0]]
            # if self.n != -1:
            #     random.shuffle(ic_egs)
            rv = []
            for egs in ic_egs:
                if egs[3]["tgt"] != test_egs[1]:
                    rv.append(egs)
            if len(rv) < 3:
                print(f"Warning: Did not find enough ic egs for a line. Proceeding with backup")
                return self.backup_data
            if self.n == -1:
                return rv
            else:
                return rv[:self.n]
        elif self.specialize == 2:
            # this is a dummy test; return all ic examples except the ones which match the tgt
            ic_egs = self.data_dict[test_egs[0]]
            # if self.n != -1:
            #     random.shuffle(ic_egs)
            rv = []
            for egs in ic_egs:
                if egs[3]["tgt"] != test_egs[1]:
                    rv.append(egs)
            rv = rv[:self.n] + self.backup_data
            return rv
        elif self.specialize == 3:
            # seg-by-seg doc level egs
            # this is a dummy test; return all ic examples except the ones which match the tgt
            ic_egs = self.data_dict[test_egs[0]]
            # if self.n != -1:
            #     random.shuffle(ic_egs)
            rv = []
            for egs in ic_egs:
                if egs[3]["tgt"] != test_egs[1]:
                    rv.append(egs)
            rv = rv[:self.n] + self.backup_data
            return rv
        elif self.specialize == 4:
            # can't match tgt, cant repeat the same error
            ic_egs = self.data_dict[test_egs[0]]
            # if self.n != -1:
            #     random.shuffle(ic_egs)
            rv = []
            error_set = []
            for egs in ic_egs:
                error_key = ','.join(str(d) for d in egs[3]["errors"])
                if egs[3]["tgt"] != test_egs[1] and error_key not in error_set:
                    rv.append(egs)
                    error_set.append(error_key)
            rv = rv[:self.n] + self.backup_data
            return rv
        else:
            random.shuffle(self.data)
            return self.data[:self.n]

def build_prompt(incontext_examples, source_language, target_language, source_sentence, target_sentence, reference_sentence=None):
    initial_instruct = {"role":"system", "content":"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation."
    }

    detailed_instruct_text = "\nBased on the source and target sentences surrounded with triple backticks ('''), identify error types in the translation and classify them. Please identify all errors within each translated segment, up to a maximum of five. If there are more than five errors, identify only the five most severe. The format of your output should be a json object in single line format. Directly generate this output without any additional reasoning.\nThe categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: major, minor, and neutral. Major errors inhibit comprehension of the text or disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension. No-errors should be marked as neutral.\n"

    incontext = []
    examples = incontext_examples.get_examples((source_sentence, target_sentence))
    for egs in examples:
        if isinstance(egs[0], list):
            assert len(egs[0]) == len(egs[1])
            for egs_0, egs_1 in zip(egs[0], egs[1]):
                incontext.append({"role":"user", "content": egs_0+detailed_instruct_text})
                incontext.append({"role":"assistant", "content": egs_1})
        else:
            incontext.append({"role":"user", "content": egs[0]+detailed_instruct_text})
            incontext.append({"role":"assistant", "content": egs[1]})

    if reference_sentence is not None:
        prompt = {"role":"user", "content": f"{source_language} source:\n'''{source_sentence}'''\n{target_language} translation:\n'''{target_sentence}'''\n{target_language} reference:\n'''{reference_sentence}'''\n"+detailed_instruct_text}
    else:
        prompt = {"role":"user", "content": f"{source_language} source:\n'''{source_sentence}'''\n{target_language} translation:\n'''{target_sentence}'''\n"+detailed_instruct_text}

    messages = [initial_instruct] + incontext + [prompt]
    return messages