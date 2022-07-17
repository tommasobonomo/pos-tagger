import logging
from pathlib import Path
from typing import List, Tuple

import requests

from settings import data_dir, extension

logging.basicConfig(level=logging.INFO)
console_logger = logging.getLogger("download_and_parse_data")

SPLITS = ["train", "dev", "test"]

Word = str
Label = str
Row = Tuple[Word, Label]
Sentence = List[Row]


def get_remote_files_urls() -> Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]]:
    BASE_PATH = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{placeholder}.conllu"
    return (
        ("train", BASE_PATH.format(placeholder="train")),
        ("dev", BASE_PATH.format(placeholder="dev")),
        ("test", BASE_PATH.format(placeholder="test")),
    )


def parse_conllu(full_text: str) -> List[Sentence]:
    lines = full_text.splitlines()
    parsed_sentences = []
    sentence = []
    for line in lines:
        cells = line.split("\t")
        # CoNLL-U format gives an index to each word, so we care only about lines with a number as the first character
        if cells[0].isnumeric():
            # All interesting values are separated by a tab character, so we split on that
            # We are interested only in the word and the Universal POS tag, which are respectively at columns 1 and 3
            word, pos_label = cells[1], cells[3]
            sentence.append((word, pos_label))
        elif "-" in cells[0]:
            # It's a multi-token word, ignore it and use the tokenized version that follows
            continue
        else:
            # We need to add the sentence to the List of parsed sentences, only if the sentence is actually something
            if len(sentence) > 0:
                parsed_sentences.append(sentence)
                sentence = []

    # Add any sentence that remains in the accumulator
    if sentence:
        parsed_sentences.append(sentence)

    return parsed_sentences


def save_sentences_to_file(sentences: List[Sentence], output_path: Path):
    with open(output_path, "w+") as f:
        for sentence in sentences:
            for word, tag in sentence:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")


def main():
    # Define file paths
    file_paths = [data_dir / f"{split}{extension}" for split in SPLITS]

    # Check if data directory and splits have already been retrieved
    if all(file_path.exists() for file_path in file_paths):
        console_logger.info("All datafiles already retrieved, done ✅")
        return

    # There are some files missing, redownload all of them
    data_dir.mkdir(exist_ok=True)
    remote_file_urls = get_remote_files_urls()
    for split, url in remote_file_urls:
        console_logger.info(f"Downloading {split} split..")
        response = requests.get(url)

        # Parse sentences from downloaded file
        console_logger.info("Parsing split...")
        split_sentences = parse_conllu(response.text)

        # Save to given path
        output_path = data_dir / f"{split}{extension}"
        console_logger.info(f"Saving {split} split to path {str(output_path)}...")
        save_sentences_to_file(split_sentences, output_path)

        console_logger.info(f"Finished downloading {split} split.")

    console_logger.info("Finished all splits ✅")


if __name__ == "__main__":
    main()
